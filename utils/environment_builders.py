"""Single-environment builder helpers."""

from __future__ import annotations

import re

import gymnasium as gym

from utils.environment_types import get_env_type


def merge_env_kwargs(env_kwargs: dict, defaults: dict) -> dict:
    """Merge env kwargs with defaults without mutating the input."""
    merged = dict(env_kwargs)
    for key, value in defaults.items():
        merged.setdefault(key, value)
    return merged


def annotate_vec_env(vec_env, env_id, env_spec, obs_type, render_mode, project_id, seed):
    """Annotate vec env with metadata for VecEnvInfoWrapper compatibility."""
    vec_env._spec = env_spec
    vec_env.env_id = env_id
    vec_env._project_id = project_id
    vec_env._obs_type = obs_type
    vec_env.render_mode = render_mode
    vec_env._last_seed = seed
    return vec_env


def build_alepy_env(env_id, obs_type, render_mode, **env_kwargs):
    if obs_type == "objects":
        from ocatari.core import OCAtari

        env_kwargs = merge_env_kwargs(env_kwargs, {"full_action_space": True})
        return OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)

    assert obs_type in ("ram", "rgb"), f"Unsupported obs_type for ALE: {obs_type}"
    env_kwargs = merge_env_kwargs(env_kwargs, {"full_action_space": True})
    return gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)


def build_vizdoom_env(env_id, obs_type, render_mode, **env_kwargs):
    del obs_type
    from gym_wrappers.vizdoom import VizDoomEnv

    if "scenario" not in env_kwargs:
        scenario = env_id.replace("VizDoom-", "").replace("-v0", "").replace("-v1", "").replace("-", "_")
        scenario = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", scenario).lower()
        env_kwargs["scenario"] = scenario

    return VizDoomEnv(render_mode=render_mode, **env_kwargs)


def build_stable_retro_env(env_id, obs_type, render_mode, **env_kwargs):
    del obs_type
    import retro

    game = env_id.replace("Retro/", "")
    make_kwargs = dict(env_kwargs)
    state = make_kwargs.pop("state", None)
    return retro.make(game=game, state=state, render_mode=render_mode, **make_kwargs)


def build_bandit_env(env_id, obs_type, render_mode, **env_kwargs):
    del env_id, obs_type, render_mode
    from gym_envs.mab_env import MultiArmedBanditEnv

    return MultiArmedBanditEnv(**env_kwargs)


def build_gym_env(env_id, obs_type, render_mode, **env_kwargs):
    del obs_type
    return gym.make(env_id, render_mode=render_mode, **env_kwargs)


ENV_BUILDERS = {
    "alepy": build_alepy_env,
    "vizdoom": build_vizdoom_env,
    "stable_retro": build_stable_retro_env,
    "mab": build_bandit_env,
}


def build_single_env(env_id: str, obs_type: str, render_mode: str, **kwargs):
    """Dispatch to the correct builder for a single environment instance."""
    env_type = get_env_type(env_id)
    builder = ENV_BUILDERS.get(env_type, build_gym_env)
    return builder(env_id, obs_type, render_mode, **kwargs)
