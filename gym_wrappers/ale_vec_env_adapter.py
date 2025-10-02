from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv


class AtariVecEnvAdapter(VectorEnv):
    """
    Thin adapter exposing ale-py's AtariVectorEnv with a Gymnasium VectorEnv API.

    Responsibilities:
    - Present Gymnasium-style `reset()` -> obs and `step()` -> (obs, rewards, dones, infos)
    - Auto-reset finished sub-envs and emit Gym/Monitor-like episode infos
    - Optionally enforce a time-limit with TimeLimit.truncated + terminal_observation
    - Provide a no-op `recorder(...)` context manager for compatibility
    - Expose minimal metadata helpers used in this repo (return threshold, max steps, rgb flag)
    """

    def __init__(
        self,
        *,
        ale_env,  # ale_py.vector_env.AtariVectorEnv
        env_id: str,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        # Derive single-env spaces from ale vector env
        single_obs_space = getattr(ale_env, "single_observation_space", None) or getattr(ale_env, "observation_space", None)
        single_act_space = getattr(ale_env, "single_action_space", None) or getattr(ale_env, "action_space", None)
        assert isinstance(single_obs_space, spaces.Space), "ALE vector env must expose observation space"
        assert isinstance(single_act_space, spaces.Space), "ALE vector env must expose action space"

        super().__init__(num_envs=int(getattr(ale_env, "num_envs", 1)), observation_space=single_obs_space, action_space=single_act_space)

        self._ale = ale_env
        self.env_id = env_id
        self.render_mode = render_mode
        self._max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None

        # Per-env episode accounting
        self._ep_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self._ep_lengths = np.zeros((self.num_envs,), dtype=np.int32)

        # Last observations (needed to supply terminal_observation on truncation)
        self._last_obs: Optional[np.ndarray] = None
        self._pending_actions: Optional[np.ndarray] = None

    # ---- Optional helpers consumed by wrappers/call-sites ----
    def get_return_threshold(self) -> Optional[float]:
        # Unknown by default; EarlyStoppingCallback is a no-op when None
        return None

    def get_max_episode_steps(self) -> Optional[int]:
        return self._max_episode_steps

    def is_rgb_env(self) -> bool:
        return True

    # Spec/metadata helpers (best-effort, keep optional)
    def get_reward_range(self):  # noqa: D401
        """Unknown for ALE adapter (return None)."""
        return None

    def get_return_range(self):  # noqa: D401
        """Unknown for ALE adapter (return None)."""
        return None

    def get_spec(self):  # noqa: D401
        """No structured spec available from adapter; return None."""
        return None

    def get_action_labels(self):  # noqa: D401
        """No action label mapping provided; return None."""
        return None

    def get_render_mode(self):  # noqa: D401
        return self.render_mode

    def get_render_fps(self) -> int:  # noqa: D401
        # Use a sensible default; specific ALE envs may differ
        return 60

    @contextmanager
    def recorder(self, video_path: str, record_video: bool = True):
        """
        Context manager for video recording.

        Since AtariVecEnvAdapter doesn't support rendering, this returns a no-op
        context manager. Actual video recording for ALE native vectorization is
        handled by ALEVecVideoRecorder wrapper at the vector env level.
        """
        # No-op: video recording must be handled at wrapper level
        yield self

    # ---- Gymnasium VectorEnv interface ----
    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:  # noqa: D401
        """Seed is a no-op; prefer seeding via ale-py initialization if supported."""
        # Gymnasium VectorEnv interface returns a list per sub-env
        return [seed for _ in range(self.num_envs)]

    def reset(self) -> np.ndarray:
        # Reset all sub-envs and drop info for compatibility
        obs, _info = self._ale.reset()
        # Ensure batch-first shape (N, ...)
        assert isinstance(obs, np.ndarray), "ALE reset must return a numpy array"
        self._last_obs = obs
        # Reset episode accounting
        self._ep_returns.fill(0.0)
        self._ep_lengths.fill(0)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        # Cache actions to be applied in step_wait
        self._pending_actions = np.asarray(actions)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        assert self._pending_actions is not None, "step_async must be called before step_wait"
        actions = self._pending_actions
        self._pending_actions = None

        # Step underlying ALE vector env (Gymnasium v0.28-style return)
        obs_next, rewards, terminations, truncations, infos = self._ale.step(actions)

        # Convert to numpy arrays with expected dtypes
        obs_next = np.asarray(obs_next)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)

        # Optional external time-limit enforcement
        if self._max_episode_steps is not None:
            # Enforce step-limit by marking truncations when length reaches max
            reaching_limit = self._ep_lengths + 1 >= self._max_episode_steps
            truncations = np.logical_or(truncations, reaching_limit)

        # Done when either terminated or truncated
        dones = np.logical_or(terminations, truncations)

        # Update episode accounting BEFORE auto-resets
        self._ep_returns += rewards
        self._ep_lengths += 1

        # Build infos as a list of dicts
        info_list: List[dict] = [{} for _ in range(self.num_envs)]
        done_indices = np.flatnonzero(dones)
        if done_indices.size > 0:
            # Prepare reset mask for partial resets (prefer ale-py's reset_mask option)
            reset_mask = np.zeros((self.num_envs, 1), dtype=np.bool_)
            reset_mask[done_indices, 0] = True

            # Emit Gym/Monitor-like episode summaries
            for idx in done_indices:
                info = info_list[int(idx)]
                info["episode"] = {"r": float(self._ep_returns[idx]), "l": int(self._ep_lengths[idx])}
                # If this is a time-limit truncation, emit TimeLimit-specific fields
                if bool(truncations[idx]) and not bool(terminations[idx]):
                    info["TimeLimit.truncated"] = True
                    # terminal_observation is the observation at the final step
                    # We use the just-produced next observation prior to auto-reset
                    info["terminal_observation"] = np.array(obs_next[idx])

            # Perform partial reset on done envs so returned obs are start-of-episode
            reset_done = False
            new_obs_batch: Optional[np.ndarray] = None
            try:
                # Newer ale-py API supports mask via reset(options={"reset_mask": ...})
                _res = self._ale.reset(options={"reset_mask": reset_mask})
                if isinstance(_res, tuple):
                    new_obs_batch = _res[0]
                elif isinstance(_res, np.ndarray):
                    new_obs_batch = _res
                reset_done = True
            except TypeError:
                # Older variants may expose these helpers
                try:
                    self._ale.reset(np.flatnonzero(dones))  # type: ignore[arg-type]
                    reset_done = True
                except Exception:
                    try:
                        self._ale.reset_done(dones)  # type: ignore[attr-defined]
                        reset_done = True
                    except Exception:
                        reset_done = False

            if reset_done and isinstance(new_obs_batch, np.ndarray):
                # Overwrite done indices with the first observations of the new episodes
                obs_next[done_indices] = new_obs_batch[done_indices]

            # Clear per-episode counters for reset envs
            self._ep_returns[done_indices] = 0.0
            self._ep_lengths[done_indices] = 0

        # Track last obs for potential future use
        self._last_obs = obs_next

        return obs_next, rewards, dones, info_list

    # Convenience path used by some callers/wrappers
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:  # type: ignore[override]
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        try:
            self._ale.close()
        except Exception:
            pass

    # -------- Extension points used by wrappers --------
    def env_method(self, method_name: str, *args, indices: Optional[List[int]] = None, **kwargs) -> List[Any]:  # noqa: D401
        """Call methods on this adapter.

        VecEnvInfoWrapper and VecVideoRecorder use `env_method(..., indices=[0])` to
        fetch helper methods from the first underlying env. Since ale-py's vector env
        does not expose per-env Python objects, we surface the helpers on the adapter
        itself and return a single-element list with the result.
        """
        target = getattr(self, method_name, None)
        if not callable(target):
            raise AttributeError(f"Method '{method_name}' not found on ALE adapter")
        return [target(*args, **kwargs)]

    # Abstract helper methods ---------------------------------------------------
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:  # noqa: D401
        """Return attribute values for each selected sub-env (adapter-wide)."""
        value = getattr(self, attr_name, None)
        n = len(indices) if indices is not None else self.num_envs
        return [value for _ in range(n)]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:  # noqa: D401
        """Set an attribute on the adapter (no per-sub-env granularity)."""
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices: Optional[List[int]] = None) -> List[bool]:  # noqa: D401
        """Report wrapper presence (adapter is not wrapped by vectorized wrappers)."""
        n = len(indices) if indices is not None else self.num_envs
        return [False for _ in range(n)]

    # Optional render hook ---------------------------------------------------
    def render(self, mode: str = "human"):
        return None
