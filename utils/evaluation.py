"""Evaluation utilities for running policies on vectorized environments.

Provides evaluate_policy that runs exactly N episodes across a vectorized
environment, balancing the number of episodes per environment instance to
avoid bias (distributes as evenly as possible when N is not a multiple of
num_envs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from .torch import _device_of, inference_ctx
from .policy_ops import policy_act


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
    max_steps_per_episode: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
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
    import time
    obs = env.reset()
    per_env_targets = _balanced_targets(n_envs, int(n_episodes))
    per_env_counts = [0] * n_envs
    per_env_rewards: List[List[float]] = [[] for _ in range(n_envs)]
    per_env_lengths: List[List[int]] = [[] for _ in range(n_envs)]
    # Track ongoing episode stats per env to support hard step caps
    cur_rewards = [0.0] * n_envs
    cur_lengths = [0] * n_envs

    total_timesteps = 0  # counts transitions across all envs
    start_time = time.time()

    with inference_ctx(policy_model):
        while True:
            # Stop when all envs reached their target
            if all(count >= tgt for count, tgt in zip(per_env_counts, per_env_targets)):
                break

            obs_t = torch.as_tensor(obs, device=device)
            actions_t, _, _ = policy_act(policy_model, obs_t, deterministic=deterministic)
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
                    # Use our running counters as a last resort
                    per_env_rewards[idx].append(float(cur_rewards[idx]))
                    per_env_lengths[idx].append(int(cur_lengths[idx]))
                else:
                    per_env_rewards[idx].append(float(ep.get("r", 0.0)))
                    per_env_lengths[idx].append(int(ep.get("l", 0)))
                per_env_counts[idx] += 1
                # Reset trackers for that env
                cur_rewards[idx] = 0.0
                cur_lengths[idx] = 0

            # Update running counters
            # rewards/dones may be scalars for non-vector envs (n_envs==1)
            for i in range(n_envs):
                r = float(rewards if np.isscalar(rewards) else rewards[i])
                cur_rewards[i] += r
                cur_lengths[i] += 1

            # Enforce hard cap per episode when requested
            if max_steps_per_episode is not None and max_steps_per_episode > 0:
                for i in range(n_envs):
                    if (
                        per_env_counts[i] < per_env_targets[i]
                        and cur_lengths[i] >= int(max_steps_per_episode)
                    ):
                        # Finalize truncated episode with running counters
                        per_env_rewards[i].append(float(cur_rewards[i]))
                        per_env_lengths[i].append(int(cur_lengths[i]))
                        per_env_counts[i] += 1
                        cur_rewards[i] = 0.0
                        cur_lengths[i] = 0
                        # Reset only this env when possible and patch next_obs
                        ob = env.env_method("reset", indices=[i])[0]
                        if np.isscalar(next_obs):
                            next_obs = ob
                        else:
                            next_obs[i] = ob

            # Enforce wall-clock timeout if set
            if timeout_seconds is not None and (time.time() - start_time) >= float(timeout_seconds):
                break

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
