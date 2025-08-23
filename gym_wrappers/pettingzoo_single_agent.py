import numpy as np
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces


class PettingZooSingleAgentWrapper(gym.Env):
    """
    Adapt a PettingZoo AEC/parallel environment to a Gymnasium-like single-agent API.

    - Assumes a single controllable agent of interest.
    - For multi-agent envs, select the first agent by default or a named agent via `agent_id`.
    - Exposes `observation_space`, `action_space`, `reset`, `step`, `close`, `render`.
    - Returns Gymnasium-style tuples: (obs, reward, done, info) for compatibility with SB3 VecEnv.
      Note: SB3 VecEnv uses Gym v0.21 style `done`; truncation/termination are merged.
    """

    def __init__(self, pz_env, agent_id: Optional[str] = None, render_mode: Optional[str] = None):
        self._env = pz_env
        self._render_mode = render_mode
        self._is_parallel = True  # default; will be validated on reset

        # Determine controlled agent
        try:
            agents = list(getattr(self._env, "agents", []) or [])
        except Exception:
            agents = []
        if not agents:
            # initialize to populate agents if needed
            try:
                self._env.reset(seed=None, options=None)
                agents = list(getattr(self._env, "agents", []) or [])
            except Exception:
                agents = []
        if agent_id is None and agents:
            agent_id = agents[0]
        self._agent_id = agent_id

        # Try to expose spaces early if available from PettingZoo mappings
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        try:
            if self._agent_id is not None:
                obs_spaces = getattr(self._env, "observation_spaces", {}) or {}
                act_spaces = getattr(self._env, "action_spaces", {}) or {}
                self.observation_space = obs_spaces.get(self._agent_id, None)
                self.action_space = act_spaces.get(self._agent_id, None)
        except Exception:
            pass

        # If spaces are still unknown, perform a lightweight reset to infer them
        if self.observation_space is None or self.action_space is None:
            try:
                _ = self.reset()
                # After reset, try again to fetch spaces
                if self._agent_id is not None:
                    obs_spaces = getattr(self._env, "observation_spaces", {}) or {}
                    act_spaces = getattr(self._env, "action_spaces", {}) or {}
                    self.observation_space = obs_spaces.get(self._agent_id, self.observation_space)
                    self.action_space = act_spaces.get(self._agent_id, self.action_space)
            except Exception:
                pass
        self._opponent_ids = []

    # --- Gymnasium-like API -------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # PettingZoo parallel_env.reset returns observations dict (and optionally infos)
        obs_dict = self._env.reset(seed=seed, options=options)  # type: ignore[call-arg]
        if isinstance(obs_dict, tuple) and len(obs_dict) == 2 and isinstance(obs_dict[0], dict):
            # Some versions may return (obs, infos)
            obs_dict, infos_dict = obs_dict
        else:
            infos_dict = {}

        # Detect agent ids and spaces now
        try:
            agents = list(getattr(self._env, "agents", []) or [])
        except Exception:
            agents = []
        if self._agent_id is None and agents:
            self._agent_id = agents[0]
        self._opponent_ids = [a for a in agents if a != self._agent_id]

        # Expose spaces
        try:
            self.observation_space = getattr(self._env, "observation_spaces", {}).get(self._agent_id)
            self.action_space = getattr(self._env, "action_spaces", {}).get(self._agent_id)
        except Exception:
            pass

        # Parallel mode assumption for Go
        self._is_parallel = True

        obs = None
        if isinstance(obs_dict, dict):
            obs = obs_dict.get(self._agent_id)
        if obs is None:
            try:
                obs = self._env.observe(self._agent_id)
                self._is_parallel = False
            except Exception:
                pass

        # Do NOT coerce dict observations to numpy arrays; SB3 expects mapping when observation_space is Dict
        if not isinstance(obs, dict):
            try:
                obs = np.asarray(obs)
            except Exception:
                pass
        # Gymnasium API: return (obs, info)
        info = {}
        try:
            if isinstance(infos_dict, dict) and self._agent_id is not None:
                info = dict(infos_dict.get(self._agent_id, {}) or {})
        except Exception:
            info = {}
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # Handle action as NumPy array scalar or Python int
        if isinstance(action, (np.ndarray,)) and action.shape == ():
            action = action.item()

        if self._is_parallel:
            # Build actions dict: our action + random opponents
            actions = {self._agent_id: action}
            # Include only active opponent agents
            active_agents = list(getattr(self._env, "agents", []) or [])
            for oid in self._opponent_ids:
                if oid in active_agents:
                    try:
                        space = getattr(self._env, "action_spaces", {}).get(oid)
                        actions[oid] = space.sample() if space is not None else 0
                    except Exception:
                        actions[oid] = 0

            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(actions)

            obs = obs_dict.get(self._agent_id)
            reward = float(rew_dict.get(self._agent_id, 0.0))
            terminated = bool(term_dict.get(self._agent_id, False))
            truncated = bool(trunc_dict.get(self._agent_id, False))
            info = dict(info_dict.get(self._agent_id, {}) or {})
            # Preserve dict observations; otherwise convert to numpy array
            if not isinstance(obs, dict):
                try:
                    obs = np.asarray(obs)
                except Exception:
                    pass
            return obs, reward, terminated, truncated, info
        else:
            # AEC fallback: step for our agent, random-step others until next turn of our agent
            reward_total = 0.0
            terminated = False
            truncated = False
            info: Dict[str, Any] = {}

            try:
                self._env.step(action)
            except Exception:
                pass

            # Advance until our agent acts again or episode end
            max_turns = 1000  # safety to avoid infinite loops
            turns = 0
            while turns < max_turns:
                turns += 1
                try:
                    term_map = getattr(self._env, "terminations", {}) or {}
                    trunc_map = getattr(self._env, "truncations", {}) or {}
                    terminated = bool(term_map.get(self._agent_id, False))
                    truncated = bool(trunc_map.get(self._agent_id, False))
                    if terminated or truncated:
                        break
                except Exception:
                    pass

                # If it's our turn again, break
                try:
                    if getattr(self._env, "agent_selection", None) == self._agent_id:
                        break
                except Exception:
                    pass

                # Otherwise step a random opponent if required
                try:
                    current_agent = getattr(self._env, "agent_selection", None)
                    if current_agent and current_agent != self._agent_id:
                        space = getattr(self._env, "action_spaces", {}).get(current_agent)
                        opp_action = space.sample() if space is not None else 0
                        self._env.step(opp_action)
                    else:
                        break
                except Exception:
                    break

            try:
                rew_map = getattr(self._env, "rewards", {}) or {}
                reward_total += float(rew_map.get(self._agent_id, 0.0))
            except Exception:
                pass

            try:
                info_map = getattr(self._env, "infos", {}) or {}
                info = dict(info_map.get(self._agent_id, {}) or {})
            except Exception:
                info = {}

            try:
                obs = self._env.observe(self._agent_id)
            except Exception:
                obs = None
            if not isinstance(obs, dict):
                try:
                    obs = np.asarray(obs)
                except Exception:
                    pass

            return obs, float(reward_total), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        try:
            return self._env.render()
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


