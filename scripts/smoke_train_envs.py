import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _discover_config_specs() -> List[Tuple[str, str]]:
    """Discover runnable config specs as (project_id, variant) pairs.

    Reuses the same discovery logic as scripts/smoke_test_envs.py which builds
    ids in the form '<project_id>_<variant>'. We split on the last underscore
    to produce (project_id, variant) pairs compatible with train.py expectations
    ('<project_id>:<variant>').
    """
    from scripts.smoke_test_envs import discover_env_config_ids  # type: ignore

    config_dir = PROJECT_ROOT / "config" / "environments"
    config_ids = discover_env_config_ids(config_dir)
    specs: List[Tuple[str, str]] = []
    for cid in config_ids:
        # Split on the last underscore to allow underscores in project_id
        if "_" not in cid:
            # Skip malformed ids defensively
            continue
        project_id, variant = cid.rsplit("_", 1)
        specs.append((project_id, variant))
    return specs


def _train_once(project_id: str, variant: str, timesteps: int, quiet: bool = True) -> Tuple[bool, str]:
    """Run a minimal training loop for a single config.

    Returns (ok, message). Configures W&B in disabled mode to avoid network
    I/O during smoke testing.
    """
    # Disable W&B networking/noise for smoke runs
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    # Build directly using project utils to allow small overrides
    from stable_baselines3.common.utils import set_random_seed
    from utils.config import load_config
    from agents import build_agent
    from contextlib import contextmanager
    from utils.train_launcher import _ensure_wandb_run_initialized

    # Runtime patches to avoid video/GUI requirements during smoke training
    @contextmanager
    def _patched_env_builders():
        from agents.base_agent import BaseAgent  # type: ignore
        from gym_wrappers.env_info import EnvInfoWrapper  # type: ignore
        from utils.environment import build_env_from_config  # type: ignore

        # 1) Disable video and rendering for val/test by overriding BaseAgent.build_env
        orig_build_env = BaseAgent.build_env

        def build_env_no_video(self, stage: str, **kwargs):
            stage_kwargs = {
                "train": {"seed": self.config.seed},
                "val": {"seed": self.config.seed + 1000, "subproc": False, "render_mode": None, "record_video": False},
                "test": {"seed": self.config.seed + 2000, "subproc": False, "render_mode": None, "record_video": False},
            }
            self._envs = self._envs if hasattr(self, "_envs") else {}
            self._envs[stage] = build_env_from_config(self.config, **{**stage_kwargs[stage], **kwargs})

        # 2) Provide a no-op recorder() context manager when present
        from contextlib import contextmanager as _cm

        @_cm
        def _noop_recorder(self, video_path: str, record_video: bool = True):
            yield self

        # Apply patches
        BaseAgent.build_env = build_env_no_video  # type: ignore[assignment]
        # Only patch when method missing to avoid shadowing real recorder
        need_recorder_patch = not hasattr(EnvInfoWrapper, "recorder")
        if need_recorder_patch:
            setattr(EnvInfoWrapper, "recorder", _noop_recorder)  # type: ignore[attr-defined]

        try:
            yield
        finally:
            # Restore original method
            BaseAgent.build_env = orig_build_env  # type: ignore[assignment]
            if need_recorder_patch:
                try:
                    delattr(EnvInfoWrapper, "recorder")
                except Exception:
                    pass

    try:
        with _patched_env_builders():
            # Load config and apply minimal overrides for speed/stability
            cfg = load_config(project_id, variant)
            cfg.quiet = bool(quiet)
            cfg.max_timesteps = int(timesteps)
            # Keep small rollouts and a single update for speed
            cfg.n_envs = 1
            cfg.n_steps = min(int(timesteps), 64)
            cfg.n_epochs = 1
            # Ensure minibatches divide rollout exactly
            cfg.batch_size = cfg.n_envs * cfg.n_steps
            # Avoid schedule machinery during smoke runs
            cfg.policy_lr_schedule = None
            cfg.ent_coef_schedule = None
            cfg.vf_coef_schedule = None
            cfg.clip_range_schedule = None
            # Minimize eval overhead
            cfg.eval_warmup_epochs = 0
            cfg.eval_freq_epochs = None
            cfg.eval_episodes = 1
            # CPU to avoid accidental GPU/MPS usage in CI
            cfg.accelerator = cfg.AcceleratorType.cpu  # type: ignore[attr-defined]
            cfg.devices = 1

            # Ensure reproducibility and initialize a (disabled) wandb run
            set_random_seed(cfg.seed)
            _ensure_wandb_run_initialized(cfg)

            # Build and train
            agent = build_agent(cfg)
            agent.learn()
        return True, "ok"
    except SystemExit as e:
        # Normalize SystemExit from launcher argument validation
        return False, f"SystemExit: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-train all environment configs and report pass/fail.")
    parser.add_argument("--timesteps", type=int, default=100, help="Max timesteps to train per config (default: 100)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of configs to run")
    parser.add_argument("--filter", type=str, default=None, help="Only run configs containing this substring")
    parser.add_argument("--no-quiet", action="store_true", help="Run with interactive prompts enabled")
    args = parser.parse_args()

    specs = _discover_config_specs()
    if args.filter:
        specs = [(p, v) for (p, v) in specs if args.filter in p or args.filter in v]
    if args.limit is not None:
        specs = specs[: int(args.limit)]

    if not specs:
        print("No runnable environment configs found under config/environments.")
        return 1

    print(f"Discovered {len(specs)} runnable configs.")

    results: List[Tuple[str, str, bool, str]] = []
    for idx, (project_id, variant) in enumerate(specs, start=1):
        label = f"{project_id}:{variant}"
        print(f"[{idx}/{len(specs)}] {label}: training {args.timesteps} steps ...", end=" ")
        ok, msg = _train_once(project_id, variant, args.timesteps, quiet=not args.no_quiet)
        results.append((project_id, variant, ok, msg))
        print("PASS" if ok else f"FAIL ({msg})")

    # Summary
    failed = [item for item in results if not item[2]]
    print("\n=== Smoke Train Summary ===")
    print(f"Total: {len(results)}    Passed: {len(results) - len(failed)}    Failed: {len(failed)}")
    print("Results:")
    for p, v, ok, msg in results:
        status = "✅" if ok else "❌"
        suffix = "" if ok else f" — {msg}"
        print(f"- {status} {p}:{v}{suffix}")

    # Non-zero when failures present to allow CI to catch regressions
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
