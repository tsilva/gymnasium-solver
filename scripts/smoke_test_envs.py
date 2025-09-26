import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.io import read_yaml

# Ensure project root is on sys.path so `utils` and other local modules import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _collect_env_config_map(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return mapping of config_id -> raw config dict supporting legacy and new styles.

    - Legacy: top-level mapping of ids -> mapping (with optional 'inherits')
    - New: base fields at root + variant sections (e.g., ppo: {...}) create ids '<project>_<variant>'
    Skips files ending with .new.yaml.
    """
    all_configs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        if path.name.endswith(".new.yaml"):
            continue
        try:
            doc = read_yaml(path) or {}
            if not isinstance(doc, dict):
                continue
        except Exception:
            continue
        # New style detection: env_id at root OR under _base; dict fields present
        has_env_root = isinstance(doc.get("env_id"), str)
        has_base = isinstance(doc.get("_base"), dict)
        base_section = doc.get("_base") if has_base else {}
        if has_env_root or (has_base and isinstance(base_section.get("env_id"), str)):
            project = (
                doc.get("project_id")
                or (base_section.get("project_id") if isinstance(base_section, dict) else None)
                or path.stem.replace(".new", "")
            )
            field_names = {
                # Minimal set sufficient to disambiguate variant sections
                "env_id", "algo_id", "n_envs", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "ent_coef", "vf_coef", "policy", "policy_kwargs",
                "obs_type", "frame_stack", "grayscale_obs", "resize_obs", "env_kwargs",
                "project_id", "accelerator", "devices", "eval_freq_epochs", "eval_episodes",
                "eval_recording_freq", "eval_deterministic", "env_wrappers", "seed", "subproc",
                "spec",
            }
            base_root = {k: v for k, v in doc.items() if k in field_names}
            # allow _base to supply fields when using anchor-based style
            if isinstance(base_section, dict):
                base_root.update({k: v for k, v in base_section.items() if k in field_names})
            for k, v in doc.items():
                if k in field_names or not isinstance(v, dict):
                    continue
                # Skip meta/utility sections (e.g., _base)
                if isinstance(k, str) and k.startswith("_"):
                    continue
                cfg = dict(base_root)
                cfg.update(v)
                cfg.setdefault("algo_id", str(k))
                cid = f"{project}_{k}"
                all_configs[cid] = cfg
        else:
            # Legacy style: ids at top-level
            for k, v in doc.items():
                if isinstance(v, dict):
                    all_configs[str(k)] = v
    return all_configs


def discover_env_config_ids(config_dir: Path) -> List[str]:
    """Return all runnable config IDs defined under config/environments.

    Skips base entries (e.g., keys starting with '__') and any entries that
    do not declare an algo_id (not runnable training configs).
    """
    all_configs = _collect_env_config_map(config_dir)

    runnable_ids: List[str] = []
    for cfg_id, cfg in all_configs.items():
        if not isinstance(cfg, dict):
            continue
        if cfg_id.startswith("__"):
            continue
        # Consider runnable if algo_id is present after inheritance resolution.
        # At discovery time, use a simple heuristic: include when 'algo_id' is present now.
        if "algo_id" in cfg:
            runnable_ids.append(cfg_id)

    return sorted(runnable_ids)


def _resolve_inheritance(config_id: str, all_configs: Dict[str, Any]) -> Dict[str, Any]:
    if config_id not in all_configs:
        raise KeyError(f"Config '{config_id}' not found")

    visited = set()

    def _resolve(cid: str) -> Dict[str, Any]:
        if cid in visited:
            chain = " -> ".join(list(visited) + [cid])
            raise ValueError(f"Circular inheritance detected: {chain}")
        visited.add(cid)
        node = dict(all_configs.get(cid) or {})
        parent = node.pop("inherits", None)
        if parent is not None:
            base = _resolve(parent)
        else:
            base = {}
        base.update(node)
        return base

    return _resolve(config_id)


def run_random_steps_for_config(config_id: str, max_timesteps: int, n_envs: int) -> Tuple[bool, str]:
    """Build the environment for a given config and run random actions.

    Returns (ok, message).
    """
    try:
        # Lazy import builder only; avoid full Config validation
        from utils.environment import build_env

        # Resolve environment-centric config without strict validation
        project_root = PROJECT_ROOT
        config_dir = project_root / "config" / "environments"
        all_configs = _collect_env_config_map(config_dir)
        cfg = _resolve_inheritance(config_id, all_configs)

        # Keep environment vectorization minimal for a smoke test
        n_envs = max(1, int(n_envs))

        env = build_env(
            cfg.get("env_id"),
            n_envs=n_envs,
            seed=int(cfg.get("seed", 42)),
            project_id=cfg.get("project_id"),
            spec=cfg.get("spec"),
            env_wrappers=list(cfg.get("env_wrappers", []) or []),
            normalize_obs=cfg.get("normalize_obs", cfg.get("normalize", False) or False),
            frame_stack=int(cfg.get("frame_stack", 1) or 1),
            obs_type=cfg.get("obs_type", None),
            render_mode=None,  # avoid video/render requirements during smoke test
            subproc=False,     # keep things simple for quick checks
            record_video=False,
            env_kwargs=dict(cfg.get("env_kwargs", {}) or {}),
        )

        try:
            # Reset the vectorized env
            env.reset()

            total_steps = 0
            while total_steps < max_timesteps:
                # For VecEnv, pass a list/array of actions of length env.num_envs
                try:
                    num_envs = int(getattr(env, "num_envs", 1))
                except Exception:
                    num_envs = 1

                action_space = getattr(env, "action_space", None)
                if action_space is None:
                    raise RuntimeError("Environment missing action_space")

                # VecEnv.step expects a batch of actions with length == num_envs,
                # even when num_envs == 1. Always provide a per-env list.
                if num_envs <= 1:
                    action = [action_space.sample()]
                else:
                    action = [action_space.sample() for _ in range(num_envs)]

                obs, rewards, dones, infos = env.step(action)  # type: ignore[arg-type]
                # SB3 VecEnv auto-resets done envs internally; no manual reset needed
                total_steps += int(num_envs)

        finally:
            try:
                env.close()
            except Exception:
                pass

        return True, "ok"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test all environment configurations by running random steps.")
    parser.add_argument("--timesteps", type=int, default=200, help="Total timesteps to run per config (default: 200)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs to use per config (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of configs to test")
    parser.add_argument("--filter", type=str, default=None, help="Only test config IDs containing this substring")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config" / "environments"

    config_ids = discover_env_config_ids(config_dir)
    if args.filter:
        config_ids = [cid for cid in config_ids if args.filter in cid]
    if args.limit is not None:
        config_ids = config_ids[: int(args.limit)]

    print(f"Discovered {len(config_ids)} runnable configs under {config_dir}.")
    if not config_ids:
        print("No runnable configs found.")
        return 1

    results: List[Tuple[str, bool, str]] = []
    for idx, cfg_id in enumerate(config_ids, start=1):
        print(f"[{idx}/{len(config_ids)}] {cfg_id}: running {args.timesteps} timesteps ...", end=" ")
        ok, msg = run_random_steps_for_config(cfg_id, args.timesteps, args.n_envs)
        results.append((cfg_id, ok, msg))
        print("PASS" if ok else f"FAIL ({msg})")

    # Summary
    failed = [(cid, msg) for cid, ok, msg in results if not ok]
    print("\n=== Smoke Test Summary ===")
    print(f"Total: {len(results)}    Passed: {len(results) - len(failed)}    Failed: {len(failed)}")
    if failed:
        print("Failures:")
        for cid, msg in failed:
            print(f"- {cid}: {msg}")

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
