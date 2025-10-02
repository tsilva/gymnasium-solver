"""
Rollouts module - re-exports components from refactored modules for backward compatibility.

This module has been split into:
- returns_advantages.py: Return and advantage computation functions
- rollout_stats.py: Statistics tracking classes (RollingWindow, RunningStats)
- rollout_buffer.py: Buffer storage and trajectory data structures
- rollout_collector.py: Main rollout collection logic
"""

# Re-export from returns_advantages
from utils.returns_advantages import (
    _build_idx_map_from_valid_mask,
    _build_valid_mask_and_index_map,
    _non_terminal_float_mask,
    _normalize_advantages,
    _normalize_returns,
    _real_terminal_mask,
    compute_batched_gae_advantages_and_returns,
    compute_batched_mc_returns,
    convert_returns_to_full_episode,
)

# Re-export from rollout_stats
from utils.rollout_stats import RollingWindow, RunningStats

# Re-export from rollout_buffer
from utils.rollout_buffer import (
    RolloutBuffer,
    RolloutTrajectory,
    _flat_env_major,
    _to_np,
)

# Re-export from rollout_collector
from utils.rollout_collector import RolloutCollector

__all__ = [
    # Returns and advantages
    "_build_idx_map_from_valid_mask",
    "_build_valid_mask_and_index_map",
    "_non_terminal_float_mask",
    "_normalize_advantages",
    "_normalize_returns",
    "_real_terminal_mask",
    "compute_batched_gae_advantages_and_returns",
    "compute_batched_mc_returns",
    "convert_returns_to_full_episode",
    # Stats
    "RollingWindow",
    "RunningStats",
    # Buffer
    "RolloutBuffer",
    "RolloutTrajectory",
    "_flat_env_major",
    "_to_np",
    # Collector
    "RolloutCollector",
]
