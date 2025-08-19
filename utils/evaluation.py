"""Evaluation utilities for running policies on vectorized environments.

Provides evaluate_policy that runs exactly N episodes across a vectorized
environment, balancing the number of episodes per environment instance to
avoid bias (distributes as evenly as possible when N is not a multiple of
num_envs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .torch import _device_of, inference_ctx


def _balanced_targets(n_envs: int, total_episodes: int) -> List[int]:
    """Distribute total_episodes across n_envs as evenly as possible.

    Example: n_envs=3, total=10 -> [4, 3, 3]
    """
    if total_episodes <= 0:
        return [0] * n_envs
    base = total_episodes // n_envs
    rem = total_episodes % n_envs
    return [base + (1 if i < rem else 0) for i in range(n_envs)]


@torch.inference_mode()
def evaluate_policy(
    env,
    policy_model: torch.nn.Module,
    *,
    n_episodes: int,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate a policy on a (vectorized) env for exactly n_episodes.

    - Resets the env and steps until the per-env episode targets are reached
      (balanced across envs to avoid bias).
    - Relies on SB3 Monitor-provided info['episode'] for episode returns/lengths.

    Returns a metrics dict with at least:
      - total_episodes, total_timesteps, ep_rew_mean, ep_len_mean
      - per-env counts and per-env means (prefixed with per_env/*) for debugging
    """
    assert hasattr(env, "num_envs"), "Environment must be vectorized (have num_envs)"
    n_envs = int(env.num_envs)
    device = _device_of(policy_model)

    # Reset env and set up counters
    obs = env.reset()
    per_env_targets = _balanced_targets(n_envs, int(n_episodes))
    per_env_counts = [0] * n_envs
    per_env_rewards: List[List[float]] = [[] for _ in range(n_envs)]
    per_env_lengths: List[List[int]] = [[] for _ in range(n_envs)]

    total_timesteps = 0  # counts transitions across all envs

    with inference_ctx(policy_model):
        while True:
            # Stop when all envs reached their target
            if all(count >= tgt for count, tgt in zip(per_env_counts, per_env_targets)):
                break

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions_t, _, _ = policy_model.act(obs_t, deterministic=deterministic)
            actions = actions_t.detach().cpu().numpy()

            next_obs, rewards, dones, infos = env.step(actions)
            total_timesteps += n_envs

            # Process completed episodes using info['episode']
            done_idxs = np.where(dones)[0]
            for idx in done_idxs:
                if per_env_counts[idx] >= per_env_targets[idx]:
                    # Ignore extra episodes for envs that already reached target
                    continue
                info = infos[idx]
                ep = info.get("episode")
                if ep is None:
                    # Fallback: accumulate from arrays if monitor info missing
                    # Note: this branch should rarely trigger when using SB3 make_vec_env
                    # We conservatively skip in that case to avoid incorrect accounting.
                    continue
                per_env_rewards[idx].append(float(ep.get("r", 0.0)))
                per_env_lengths[idx].append(int(ep.get("l", 0)))
                per_env_counts[idx] += 1

            obs = next_obs

    # Aggregate metrics
    all_rewards = [r for env_rs in per_env_rewards for r in env_rs]
    all_lengths = [l for env_ls in per_env_lengths for l in env_ls]

    total_episodes_collected = int(sum(per_env_counts))
    ep_rew_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
    ep_len_mean = float(np.mean(all_lengths)) if all_lengths else 0.0

    # Per-env summaries (useful for debugging balancing)
    per_env_reward_means = [float(np.mean(rs)) if rs else 0.0 for rs in per_env_rewards]
    per_env_length_means = [float(np.mean(ls)) if ls else 0.0 for ls in per_env_lengths]

    metrics = {
        "total_episodes": total_episodes_collected,
        "total_timesteps": int(total_timesteps),
        "ep_rew_mean": ep_rew_mean,
        "ep_len_mean": float(ep_len_mean),
    }
    # Add per-env diagnostics
    for i in range(n_envs):
        metrics[f"per_env/episodes_{i}"] = int(per_env_counts[i])
        metrics[f"per_env/ep_rew_mean_{i}"] = per_env_reward_means[i]
        metrics[f"per_env/ep_len_mean_{i}"] = per_env_length_means[i]

    return metrics
