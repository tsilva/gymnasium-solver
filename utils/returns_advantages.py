from typing import Optional, Tuple

import numpy as np


def _real_terminal_mask(dones: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    """Real terminal if done and not a time-limit truncation."""
    dones_b = np.asarray(dones, dtype=bool)
    timeouts_b = np.asarray(timeouts, dtype=bool)
    return np.logical_and(dones_b, ~timeouts_b)


def _non_terminal_float_mask(dones: np.ndarray, timeouts: Optional[np.ndarray]) -> np.ndarray:
    """Float32 mask of non-terminals given dones/timeouts."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    return (~real_terminal).astype(np.float32)


def _build_idx_map_from_valid_mask(valid_mask_flat: np.ndarray) -> Optional[np.ndarray]:
    """Map each position to the nearest previous valid index; None if no valid entries."""
    valid_mask_flat = np.asarray(valid_mask_flat, dtype=bool)
    if not valid_mask_flat.any():
        return None
    N = int(valid_mask_flat.size)
    idxs = np.arange(N, dtype=np.int64)
    cur = np.where(valid_mask_flat, idxs, -1)
    filled = np.maximum.accumulate(cur)
    first_valid = int(np.argmax(valid_mask_flat))
    filled[filled < 0] = first_valid
    return filled


def _build_valid_mask_and_index_map(
    dones: np.ndarray,
    timeouts: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build env-major valid mask up to last real terminal and its index map."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    if real_terminal.size == 0:
        return None, None
    T, n_envs = real_terminal.shape
    valid_mask_2d = np.zeros((T, n_envs), dtype=bool)
    for j in range(n_envs):
        term_idxs = np.where(real_terminal[:, j])[0]
        if term_idxs.size > 0:
            last_term = int(term_idxs[-1])
            valid_mask_2d[: last_term + 1, j] = True
    valid_mask_flat = valid_mask_2d.transpose(1, 0).reshape(-1)
    if not valid_mask_flat.any():
        return None, None
    idx_map = _build_idx_map_from_valid_mask(valid_mask_flat)
    return valid_mask_flat, idx_map


def _normalize_returns(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return return array normalized across all elements with epsilon stability."""
    ret_flat = returns.reshape(-1)
    return (returns - ret_flat.mean()) / (ret_flat.std() + float(eps))


def _normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return advantage array normalized across all elements with epsilon stability."""
    adv_flat = advantages.reshape(-1)
    return (advantages - adv_flat.mean()) / (adv_flat.std() + float(eps))


def compute_batched_mc_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted Monte Carlo-style returns for batched trajectories."""

    T, n_envs = rewards.shape
    returns_buf = np.zeros_like(rewards, dtype=np.float32)
    returns_acc = np.zeros(n_envs, dtype=np.float32)

    # Create a mask of non-terminal observations
    # (all observations that are not done and not timeout)
    non_terminal = _non_terminal_float_mask(dones, timeouts)

    for t in range(T - 1, -1, -1):
        _rewards_t = rewards[t]
        _non_terminal_t = non_terminal[t]
        _future_returns = returns_acc * _non_terminal_t
        _discounted_future_returns = gamma * _future_returns
        returns_acc = _rewards_t + _discounted_future_returns
        returns_buf[t] = returns_acc

    return returns_buf

def convert_returns_to_full_episode(
    returns: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    """Convert reward-to-go returns to per-episode constants in-place."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    T, n_envs = returns.shape if returns.size > 0 else (0, 0)
    for j in range(n_envs):
        seg_start = 0
        seg_value = returns[seg_start, j] if T > 0 else 0.0
        for t in range(T):
            # Set the return at step t to the segment's initial return
            returns[t, j] = seg_value
            if not real_terminal[t, j]:
                continue
            seg_start = t + 1
            if seg_start >= T:
                break
            seg_value = returns[seg_start, j]
    return returns

def compute_batched_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
    last_values: np.ndarray,
    bootstrapped_next_values: Optional[np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute batched GAE(λ) advantages and returns for (T, N) rollouts."""
    values = np.asarray(values, dtype=np.float32) # TODO: why this?
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    timeouts = np.asarray(timeouts, dtype=bool)
    last_values = np.asarray(last_values, dtype=np.float32)

    T, n_envs = rewards.shape

    # next_values: shift values by one step in time dimension; last row uses last_values
    next_values = np.zeros_like(values, dtype=np.float32)
    if T > 1: next_values[:-1] = values[1:]
    next_values[-1] = last_values

    # If bootstrapped_next_values provided, override next_values at timeout steps
    if bootstrapped_next_values is not None:
        bootstrapped_next_values = np.asarray(bootstrapped_next_values, dtype=np.float32)
        next_values = np.where(timeouts, bootstrapped_next_values, next_values)

    # Real terminals are env terminations that are not timeouts
    non_terminal = _non_terminal_float_mask(dones, timeouts)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(n_envs, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * non_terminal[t] - values[t]
        gae = delta + gamma * gae_lambda * gae * non_terminal[t]
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
