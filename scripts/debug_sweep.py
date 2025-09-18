import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict


def run_single_trial(
    config_id: str = "CartPole-v1:ppo",
    overrides: Dict[str, Any] | None = None,
    use_offline: bool = True,
    quiet: bool = True,
) -> int:
    """Run a single sweep-like trial locally for debugging.

    This sets environment variables so that train.py runs in sweep-merge mode
    and applies overrides via WANDB_SWEEP_OVERRIDES without needing a W&B Agent.
    """
    env = os.environ.copy()

    # Make wandb behave offline unless explicitly disabled
    if use_offline and "WANDB_MODE" not in env:
        env["WANDB_MODE"] = "offline"

    # Signal train.py to merge wandb.config into the dataclass
    env.setdefault("WANDB_SWEEP_ID", "debug-local-sweep")

    # Provide JSON overrides for hyperparameters
    if overrides:
        env["WANDB_SWEEP_OVERRIDES"] = json.dumps(overrides)

    args = [sys.executable, "train.py", "--config_id", config_id, "--wandb_sweep"]
    if quiet:
        args.append("-q")

    print("Launching:", " ".join(shlex.quote(a) for a in args))
    return subprocess.call(args, env=env, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> int:
    # Sensible defaults for CartPole PPO
    default_overrides = {
        "policy_lr": 3e-4,
        "clip_range": 0.2,
        "n_epochs": 10,
        "ent_coef": 0.0,
        # Fractional batch size relative to n_envs * n_steps
        "batch_size": 0.5,
        # Keep runs short for debug
        "max_timesteps": 2_000,
    }

    # Basic CLI passthrough: allow config_id via env for quick tweaks
    config_id = os.environ.get("DEBUG_SWEEP_CONFIG_ID", "CartPole-v1:ppo")
    return run_single_trial(config_id=config_id, overrides=default_overrides)


if __name__ == "__main__":
    raise SystemExit(main())


