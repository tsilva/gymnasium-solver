from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Project import path & third-party helpers
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402  (add root to PYTHONPATH)

from utils.rollouts import AsyncRolloutCollector, SyncRolloutCollector, group_trajectories_by_episode  # noqa: E402
from tsilva_notebook_utils.gymnasium import build_env as _build_env  # noqa: E402

CollectorCls = {
    "sync": SyncRolloutCollector,
    "async": AsyncRolloutCollector,
}

# -----------------------------------------------------------------------------
# Lightweight data holders
# -----------------------------------------------------------------------------

@dataclass
class SimpleConfig:
    seed: int = 0
    train_rollout_steps: int = 128

    @classmethod
    def from_args(cls, args) -> "SimpleConfig":
        return cls(
            seed=args.seed,
            train_rollout_steps=args.rollout_steps
        )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def build_env(env_id: str, seed: int, n_envs: int):
    """Delegate to the project-wide `_build_env` helper."""
    return _build_env(env_id, norm_obs=False, n_envs=n_envs, seed=seed)


def get_spaces(env):
    """Return `(obs_space, act_space)` for both vectorised & single envs."""
    if hasattr(env, "single_observation_space"):
        return env.single_observation_space, env.single_action_space
    return env.observation_space, env.action_space


def make_fc_network(in_dim: int, out_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, out_dim),
    )


def make_models(obs_space, act_space):
    if len(obs_space.shape) != 1:
        raise NotImplementedError("Only flat observation spaces are supported")
    if not isinstance(act_space, gym.spaces.Discrete):
        raise NotImplementedError("Only discrete action spaces are supported")

    obs_dim, act_dim = obs_space.shape[0], act_space.n
    return make_fc_network(obs_dim, act_dim), make_fc_network(obs_dim, 1)


def maybe_load_trained_policy(
    default_path: Path, policy_model: torch.nn.Module
) -> torch.nn.Module:
    """Load a trained policy if it exists; otherwise return the given model."""
    if False and default_path.is_file():
        try:
            policy_model = torch.load(default_path, weights_only=False)
            print(f"Loaded trained policy from {default_path}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Failed to load policy ({exc}); using random init.")
    return policy_model


# -----------------------------------------------------------------------------
# Core rollout routine – used by *both* benchmark *and* normal mode
# -----------------------------------------------------------------------------

def run_rollouts(
    cfg: SimpleConfig, # TODO: call SimpleConfig something else
    env_id: str,
    collector_kind: str,
    n_rollouts: Optional[int] = None,
    trained_policy_path: Optional[Path] = None,
    n_envs: int = 1,
) -> Dict[str, float]:
    """Collect `n_rollouts` (or infinite) and return performance statistics."""

    # Create the environment
    env = build_env(env_id, cfg.seed, n_envs)
    obs_space, act_space = get_spaces(env)

    # Create the models
    policy_model, value_model = make_models(obs_space, act_space)
    if trained_policy_path is not None: policy_model = maybe_load_trained_policy(trained_policy_path, policy_model)

    # Initialize the rollout collector
    collector_cls = CollectorCls[collector_kind]
    collector = collector_cls(cfg, env, policy_model, value_model)
    collector.start()

    total_start = time.time()
    rollout_durations: List[float] = []
    total_steps = 0
    rollout_count = 0
    try:
        # While we haven't collected enough rollouts
        while n_rollouts is None or rollout_count < n_rollouts:
            # Collect rollout, if rollout not available (async mode), restart loop and try again
            rollout_start = time.time()
            trajectories = collector.get_rollout(timeout=10.0) # TODO: adjust timeout
            if trajectories is None: continue 
            rollout_end = time.time()
            rollout_elapsed = rollout_end - rollout_start
            rollout_durations.append(rollout_elapsed)

            # Calculate stats
            steps_count = len(trajectories[0])
            total_steps += steps_count
            rollout_count += 1
            mean_step_reward = trajectories[2].mean().item()
            episodes = group_trajectories_by_episode(trajectories)
            mean_ep_reward = np.mean([sum(step[2] for step in episode) for episode in episodes])
            steps_per_second = steps_count / max(rollout_elapsed, 1e-6)
            n_episodes = len(episodes)
            
            # Log stats
            print(f"Rollout {rollout_count:04d} | steps: {steps_count:<4d} | " + " | ".join([
                f"{label}: {format(value, fmt)}"
                for label, value, fmt in [
                    ("time", rollout_elapsed, "5.2f"),
                    ("n_episodes", n_episodes, "3d"),
                    ("mean_ep_reward", mean_ep_reward, "+7.2f"),
                    ("mean_step_reward", mean_step_reward, "+7.2f"),
                    ("steps/s", steps_per_second, ".1f")
                ]
            ]))
    except KeyboardInterrupt:
        print("KeyboardInterrupt - stopping collector…")
    finally:
        # Stop collector and environment
        collector.stop()
        env.close()

    total_elapsed = time.time() - total_start
    rollout_durations_np = np.array(rollout_durations)
    stats = {
        "collector": collector_kind,
        "n_envs": n_envs,
        "steps_per_rollout": int(total_steps / max(rollout_count, 1)),
        "total_steps": total_steps,
        "total_rollouts": rollout_count,
        "total_time": total_elapsed,
        "avg_rollout_time": float(rollout_durations_np.mean()) if rollout_count else 0.0,
        "std_rollout_time": float(rollout_durations_np.std()) if rollout_count else 0.0,
        "total_throughput": total_steps / max(total_elapsed, 1e-6),
        "throughput_per_env": (total_steps / max(total_elapsed, 1e-6)) / n_envs,
    }
    print("\n=== Performance Summary ===")
    for k, v in stats.items(): print(f"{k.replace('_', ' ').title():<20}: {v}")

    return stats


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Rollout-collector benchmark & runner")
    p.add_argument("--env", default="CartPole-v1", help="Gymnasium environment ID")
    p.add_argument("--collector", choices=["sync", "async"], default="sync")
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-rollouts", type=int, default=None, help="Number of rollouts (default: infinite)")
    args = p.parse_args()

    # Load configuration
    cfg = SimpleConfig.from_args(args)

    # Load trained model (if any)
    trained_path = Path(f"./models/{args.env}_ppo_full_model_seed{args.seed}.pth")

    # Run the rollouts
    run_rollouts(
        cfg=cfg,
        env_id=args.env,
        collector_kind=args.collector,
        n_rollouts=args.n_rollouts,
        trained_policy_path=trained_path,
    )


if __name__ == "__main__":
    main()
    