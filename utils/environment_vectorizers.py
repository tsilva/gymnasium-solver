"""Vectorized environment construction helpers."""

from __future__ import annotations


from utils.environment_builders import annotate_vec_env, build_single_env
from utils.environment_types import get_env_type


def resolve_vectorization_mode(env_id: str, obs_type: str, vectorization_mode: str, n_envs: int) -> str:
    """Resolve auto vectorization to an explicit mode."""
    resolved = vectorization_mode
    if resolved == "auto" and get_env_type(env_id) == "alepy" and obs_type == "rgb":
        resolved = "alepy"
    if resolved == "auto" and get_env_type(env_id) == "stable_retro":
        resolved = "async" if n_envs > 1 else "sync"
    return resolved


def build_vec_env_alepy(
    env_id: str,
    env_spec: dict,
    env_kwargs: dict,
    env_wrappers: list,
    n_envs: int,
    vectorization_mode: str,
    seed: int,
    obs_type: str,
    render_mode: str,
    grayscale_obs: bool,
    resize_obs: tuple,
    frame_stack: int,
    frame_skip: int,
    record_video: bool,
    record_video_kwargs: dict,
    project_id: str,
):
    del env_wrappers, vectorization_mode
    from gymnasium import make_vec
    from gym_wrappers.ale_vec_video_recorder import ALEVecVideoRecorder

    assert obs_type == "rgb", "ALE native vectorization requires RGB observations"

    atari_kwargs = {}
    if grayscale_obs is not None:
        atari_kwargs["grayscale"] = grayscale_obs
    if resize_obs is not None:
        atari_kwargs["img_height"] = resize_obs[0]
        atari_kwargs["img_width"] = resize_obs[1]
    if frame_stack is not None:
        atari_kwargs["stack_num"] = frame_stack
    if frame_skip is not None:
        atari_kwargs["frameskip"] = frame_skip
    if "repeat_action_probability" in env_kwargs:
        atari_kwargs["repeat_action_probability"] = env_kwargs["repeat_action_probability"]
    atari_kwargs.setdefault("full_action_space", True)
    if "full_action_space" in env_kwargs:
        atari_kwargs["full_action_space"] = env_kwargs["full_action_space"]

    vec_env = make_vec(env_id, num_envs=n_envs, vectorization_mode=None, **atari_kwargs)
    vec_env.reset(seed=seed)
    vec_env._ale_atari_vec = True  # type: ignore[attr-defined]
    annotate_vec_env(vec_env, env_id, env_spec, obs_type, render_mode, project_id, seed)

    if record_video:
        vec_env = ALEVecVideoRecorder(vec_env, **record_video_kwargs)

    return vec_env


def build_vec_env_gym(
    env_id: str,
    env_spec: dict,
    env_kwargs: dict,
    env_wrappers: list,
    n_envs: int,
    vectorization_mode: str,
    seed: int,
    obs_type: str,
    render_mode: str,
    grayscale_obs: bool,
    resize_obs: tuple,
    frame_stack: int,
    frame_skip: int,
    record_video: bool,
    record_video_kwargs: dict,
    max_episode_steps: int,
):
    from gym_wrappers.env_info import EnvInfoWrapper
    from gym_wrappers.env_video_recorder import EnvVideoRecorder

    is_alepy_env = get_env_type(env_id) == "alepy"

    def env_fn():
        from gymnasium.wrappers import (
            AtariPreprocessing,
            FrameStackObservation,
            GrayscaleObservation,
            ResizeObservation,
            TimeLimit,
        )
        from gym_wrappers.env_wrapper_registry import EnvWrapperRegistry
        from gym_wrappers.frame_skip import FrameSkipWrapper

        local_env_kwargs = dict(env_kwargs)
        use_frameskip_wrapper = frame_skip is not None and frame_skip > 1
        atari_native_frameskip = None

        if is_alepy_env and obs_type == "rgb":
            atari_native_frameskip = 1 if use_frameskip_wrapper else (frame_skip if frame_skip is not None else 4)
            local_env_kwargs["frameskip"] = atari_native_frameskip

        env = build_single_env(env_id, obs_type, render_mode, **local_env_kwargs)

        if is_alepy_env and obs_type == "rgb":
            grayscale_flag = True if grayscale_obs is None else grayscale_obs
            screen_size = resize_obs if resize_obs is not None else 84
            if isinstance(screen_size, (tuple, list)):
                assert len(screen_size) == 2, f"resize_obs must have length 2: resize_obs={screen_size}"
                screen_size = (screen_size[1], screen_size[0])
            env = AtariPreprocessing(
                env,
                frame_skip=atari_native_frameskip or 1,
                screen_size=screen_size,
                grayscale_obs=grayscale_flag,
                grayscale_newaxis=False,
                scale_obs=False,
            )
            if use_frameskip_wrapper:
                env = FrameSkipWrapper(env, skip=frame_skip)
            for wrapper in env_wrappers:
                env = EnvWrapperRegistry.apply(env, wrapper)
            if frame_stack is not None and frame_stack > 1:
                env = FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
        else:
            if resize_obs:
                env = ResizeObservation(env, shape=resize_obs)
            if grayscale_obs:
                env = GrayscaleObservation(env, keep_dim=False)
            if use_frameskip_wrapper:
                env = FrameSkipWrapper(env, skip=frame_skip)
            for wrapper in env_wrappers:
                env = EnvWrapperRegistry.apply(env, wrapper)
            if frame_stack is not None and frame_stack > 1:
                env = FrameStackObservation(env, stack_size=frame_stack)

        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        env = EnvInfoWrapper(env, obs_type=obs_type, project_id=env_id, spec=env_spec)
        if record_video:
            env = EnvVideoRecorder(env, **record_video_kwargs)
        return env

    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    use_async = vectorization_mode == "async"
    vec_env_cls = AsyncVectorEnv if use_async else SyncVectorEnv
    vector_kwargs = {"context": "spawn"} if use_async else {}
    env_fns = [lambda i=i: (e := env_fn(), e.reset(seed=seed + i), e)[2] for i in range(n_envs)]
    return vec_env_cls(env_fns, **vector_kwargs)
