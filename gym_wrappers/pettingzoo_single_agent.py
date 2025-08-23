import numpy as np
from typing import Any, Dict, Optional, Tuple


class PettingZooSingleAgentWrapper:
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

        # Placeholders; will be set on first reset
        self.observation_space = None
        self.action_space = None
        self._opponent_ids = []

    # --- Gymnasium-like API -------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # PettingZoo parallel_env.reset returns observations dict
        obs_dict = self._env.reset(seed=seed, options=options)  # type: ignore[call-arg]
        if isinstance(obs_dict, tuple) and len(obs_dict) == 2 and isinstance(obs_dict[0], dict):
            # Some versions may return (obs, infos)
            obs_dict = obs_dict[0]

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

        try:
            obs = np.asarray(obs)
        except Exception:
            pass
        return obs

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
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
            done = bool((term_dict.get(self._agent_id, False)) or (trunc_dict.get(self._agent_id, False)))
            info = dict(info_dict.get(self._agent_id, {}) or {})
            try:
                obs = np.asarray(obs)
            except Exception:
                pass
            return obs, reward, done, info
        else:
            # AEC fallback: step for our agent, random-step others until next turn of our agent
            reward_total = 0.0
            done = False
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
                    done = bool(term_map.get(self._agent_id, False) or trunc_map.get(self._agent_id, False))
                    if done:
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
            try:
                obs = np.asarray(obs)
            except Exception:
                pass

            return obs, float(reward_total), bool(done), info

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


