"""End-of-training report callback.

Writes a human- and LLM-friendly Markdown report at the end of training
summarizing config, key metrics, checkpoints, and suggested prompt context.
Output path: <run_dir>/report.md
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl

class EndOfTrainingReportCallback(pl.Callback):
    def __init__(self, *, filename: str = "report.md"):
        super().__init__()
        self.filename = filename

    # Helper: safe numeric conversion
    @staticmethod
    def _is_number(x: Any) -> bool:
        try:
            import numbers
            return isinstance(x, numbers.Number)
        except Exception:
            return False

    @staticmethod
    def _last_non_null(values):
        for v in reversed(values):
            if v not in (None, "", "nan"):
                try:
                    # normalize to float when numeric
                    fv = float(v)
                    return fv
                except Exception:
                    return v
        return None

    @staticmethod
    def _read_metrics_csv(path: Path) -> Tuple[Dict[str, Any], Dict[str, Tuple[float, Any]]]:
        """Read wide-form metrics.csv and compute:
        - last_values: last non-null for each column
        - best_eval: best eval/ep_rew_mean with its step (if present)
        Returns (last_values, best_map[metric] -> (best_value, step)).
        """
        last_values: Dict[str, Any] = {}
        best_map: Dict[str, Tuple[float, Any]] = {}
        if not path.exists() or path.stat().st_size == 0:
            return last_values, best_map

        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                if not rows:
                    return last_values, best_map

                # Collect last non-null per field
                for field in reader.fieldnames or []:
                    vals = [r.get(field) for r in rows]
                    last_values[field] = EndOfTrainingReportCallback._last_non_null(vals)

                # Track best eval/ep_rew_mean and train/ep_rew_mean
                for metric in ("eval/ep_rew_mean", "train/ep_rew_mean"):
                    if metric in (reader.fieldnames or []):
                        best_val = None
                        best_step = None
                        for r in rows:
                            raw = r.get(metric)
                            try:
                                val = float(raw) if raw not in (None, "") else None
                            except Exception:
                                val = None
                            if val is None:
                                continue
                            if best_val is None or val > best_val:
                                best_val = val
                                best_step = r.get("total_timesteps") or r.get("train/total_timesteps")
                                try:
                                    best_step = float(best_step) if best_step is not None else None
                                except Exception:
                                    pass
                        if best_val is not None:
                            best_map[metric] = (best_val, best_step)
        except Exception:
            # Best-effort; return what we have
            pass
        return last_values, best_map

    @staticmethod
    def _yaml_like(d: Dict[str, Any], indent: int = 0) -> str:
        """Minimal YAML-ish formatting without adding dependencies."""
        lines = []
        sp = "  " * indent
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                lines.append(f"{sp}{k}:")
                lines.append(EndOfTrainingReportCallback._yaml_like(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Resolve run directory
        try:
            run_dir = Path(pl_module.run_manager.run_dir)  # type: ignore[attr-defined]
        except Exception:
            # Fallback to local runs/@latest-run if available
            run_dir = Path("runs/@latest-run")
        run_dir.mkdir(parents=True, exist_ok=True)

        report_path = run_dir / self.filename

        # Gather config
        cfg = getattr(pl_module, "config", None)
        if is_dataclass(cfg):
            cfg_dict = asdict(cfg)
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            # Reflect non-callable attrs as last resort
            cfg_dict = {}
            for a in dir(cfg):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(cfg, a)
                    if not callable(v):
                        cfg_dict[a] = v
                except Exception:
                    continue

        # Metrics.csv
        metrics_csv = run_dir / "metrics.csv"
        last_vals, best_map = self._read_metrics_csv(metrics_csv)

        # Best checkpoints/videos
        ckpt_dir = run_dir / "checkpoints"
        best_ckpt = ckpt_dir / "best.ckpt"
        last_ckpt = ckpt_dir / "last.ckpt"
        best_mp4 = ckpt_dir / "best.mp4"
        last_mp4 = ckpt_dir / "last.mp4"

        # Duration
        try:
            fit_start_ns = getattr(pl_module, "fit_start_time", None)
            duration_sec = None
            if fit_start_ns is not None:
                # Use monotonic clock for consistency with BaseAgent timing
                duration_sec = max((time.perf_counter_ns() - float(fit_start_ns)) / 1e9, 0.0)
        except Exception:
            duration_sec = None

        # Determinations
        env_id = cfg_dict.get("env_id")
        algo_id = cfg_dict.get("algo_id")
        seed = cfg_dict.get("seed")
        n_envs = cfg_dict.get("n_envs")
        eval_freq = cfg_dict.get("eval_freq_epochs")
        reward_thr = cfg_dict.get("reward_threshold")
        best_eval = best_map.get("eval/ep_rew_mean")
        best_train = best_map.get("train/ep_rew_mean")
        reached_thr = None
        try:
            if reward_thr is not None and best_eval is not None:
                reached_thr = bool(float(best_eval[0]) >= float(reward_thr))
        except Exception:
            reached_thr = None

        # Select a few last values of interest
        def _lv(name, default=None):
            v = last_vals.get(name, default)
            return v

        key_last_values = {
            "train/total_timesteps": _lv("train/total_timesteps", _lv("total_timesteps")),
            "train/epoch": _lv("train/epoch"),
            "train/fps": _lv("train/fps"),
            "train/fps_instant": _lv("train/fps_instant"),
            "train/ep_rew_mean": _lv("train/ep_rew_mean"),
            "eval/ep_rew_mean": _lv("eval/ep_rew_mean"),
            "eval/ep_len_mean": _lv("eval/ep_len_mean"),
        }

        # Compose Markdown
        lines = []
        lines.append(f"# Run Report: {env_id} · {algo_id}")
        lines.append("")
        try:
            run_id = getattr(pl_module.run_manager, "run_id", None)
        except Exception:
            run_id = None
        date_str = time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"- Date: {date_str}")
        if run_id:
            lines.append(f"- Run ID: {run_id}")
        lines.append(f"- Run dir: {run_dir}")
        if duration_sec is not None:
            lines.append(f"- Training duration: {duration_sec:.2f}s ({duration_sec/60:.2f} min)")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        if key_last_values.get("train/total_timesteps") is not None:
            lines.append(f"- Total timesteps: {int(float(key_last_values['train/total_timesteps']))}")
        if key_last_values.get("train/epoch") is not None:
            lines.append(f"- Final epoch: {int(float(key_last_values['train/epoch']))}")
        if key_last_values.get("train/fps") is not None:
            lines.append(f"- FPS (avg): {float(key_last_values['train/fps']):.1f}")
        if key_last_values.get("train/fps_instant") is not None:
            lines.append(f"- FPS (instant): {float(key_last_values['train/fps_instant']):.1f}")
        if key_last_values.get("train/ep_rew_mean") is not None:
            lines.append(f"- Train ep_rew_mean (last): {float(key_last_values['train/ep_rew_mean']):.3f}")
        if key_last_values.get("eval/ep_rew_mean") is not None:
            lines.append(f"- Eval ep_rew_mean (last): {float(key_last_values['eval/ep_rew_mean']):.3f}")
        if best_eval is not None:
            be, step = best_eval
            step_str = f" at step {int(step)}" if step is not None else ""
            lines.append(f"- Best eval ep_rew_mean: {be:.3f}{step_str}")
        if reward_thr is not None:
            lines.append(f"- Reward threshold target: {reward_thr}")
            if reached_thr is not None:
                lines.append(f"- Threshold reached: {'yes' if reached_thr else 'no'}")
        if (ckpt_dir.exists()):
            if best_ckpt.exists():
                lines.append(f"- Best checkpoint: {best_ckpt}")
            if last_ckpt.exists():
                lines.append(f"- Last checkpoint: {last_ckpt}")
            if best_mp4.exists():
                lines.append(f"- Best video: {best_mp4}")
            if last_mp4.exists():
                lines.append(f"- Last video: {last_mp4}")
        lines.append("")

        lines.append("## Environment & Config")
        lines.append("")
        lines.append(f"- env_id: {env_id}")
        lines.append(f"- algo_id: {algo_id}")
        lines.append(f"- seed: {seed}")
        lines.append(f"- n_envs: {n_envs}")
        lines.append(f"- eval_freq_epochs: {eval_freq}")
        lines.append("")
        lines.append("### Full config (YAML)")
        lines.append("")
        try:
            lines.append("```yaml")
            lines.append(self._yaml_like(cfg_dict))
            lines.append("```")
        except Exception:
            # Fallback to JSON
            lines.append("```json")
            lines.append(json.dumps(cfg_dict, indent=2, default=str))
            lines.append("```")

        # Metrics snapshot
        lines.append("")
        lines.append("## Metrics (last known values)")
        lines.append("")
        if last_vals:
            # Keep a compact subset plus full JSON for the LLM
            lines.append("Selected:")
            for k in [
                "train/total_timesteps",
                "train/epoch",
                "train/ep_rew_mean",
                "eval/ep_rew_mean",
                "eval/ep_len_mean",
                "train/fps",
                "train/fps_instant",
                "train/loss",
                "train/policy_loss",
                "train/value_loss",
                "train/entropy_loss",
            ]:
                if k in last_vals and last_vals[k] is not None:
                    try:
                        lines.append(f"- {k}: {float(last_vals[k]):.6g}")
                    except Exception:
                        lines.append(f"- {k}: {last_vals[k]}")
            lines.append("")
            lines.append("All last values (JSON):")
            lines.append("\n```json")
            try:
                # Ensure plain types
                sanitized = {str(k): (float(v) if self._is_number(v) else v) for k, v in last_vals.items()}
            except Exception:
                sanitized = {str(k): str(v) for k, v in last_vals.items()}
            lines.append(json.dumps(sanitized, indent=2, default=str))
            lines.append("```")
        else:
            lines.append("(No metrics.csv available)")

        # LLM prompt helper
        lines.append("")
        lines.append("## LLM prompt")
        lines.append("")
        lines.append(
            "You are an RL training assistant. Based on the config and metrics above, suggest concrete hyperparameter adjustments to improve sample efficiency and/or final reward. Consider: learning rate scheduling, entropy coefficient, batch size, n_steps, clip range (for PPO), advantage normalization, evaluation cadence, and environment-specific wrappers. Prioritize changes likely to increase eval/ep_rew_mean sooner without destabilizing training. Return a short, actionable checklist with 3–7 items and brief justifications."
        )

        # Write file
        report_path.write_text("\n".join(lines), encoding="utf-8")
        