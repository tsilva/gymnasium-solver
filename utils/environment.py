import re
import gymnasium as gym
from gym_wrappers.env_wrapper_registry import EnvWrapperRegistry


def _is_alepy_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("ale/")

def _is_vizdoom_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("vizdoom-")

def _is_stable_retro_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("retro/")

def _is_mab_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("bandit-") or env_id.lower().startswith("bandit/") or env_id.lower() == "bandit-v0"

def _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs):
    if obs_type == "objects": return _build_env_alepy__objects(env_id, render_mode, **env_kwargs)
    else: return _build_env_alepy__standard(env_id, obs_type, render_mode, **env_kwargs)

def _build_env_alepy__objects(env_id, render_mode, **env_kwargs):
    from ocatari.core import OCAtari
    return OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)

def _build_env_alepy__standard(env_id, obs_type, render_mode, **env_kwargs):
    assert obs_type in ("ram", "rgb"), f"Unsupported obs_type for ALE: {obs_type}" # TODO: use enum to reference obs type
    return gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)

def _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs):
    from gym_wrappers.vizdoom import VizDoomEnv
    scenario = env_id.replace("VizDoom-", "").replace("-v0", "").replace("-v1", "").replace("-", "_")
    scenario = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', scenario).lower()
    return VizDoomEnv(
        scenario=scenario, 
        render_mode=render_mode, 
        **env_kwargs
    )

def _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs):
    import retro
    game = env_id.replace("Retro/", "")
    make_kwargs = dict(env_kwargs)
    make_kwargs.setdefault("use_restricted_actions", retro.Actions.DISCRETE)
    state = make_kwargs.pop("state", None)
    return retro.make(game=game, state=state, render_mode=render_mode, **make_kwargs)

def _build_env_mab(env_id, obs_type, render_mode, **env_kwargs):
    from gym_envs.mab_env import MultiArmedBanditEnv
    return MultiArmedBanditEnv(**env_kwargs)

def _build_env_gym(env_id, obs_type, render_mode, **env_kwargs):
    import gymnasium as gym
    return gym.make(env_id, render_mode=render_mode, **env_kwargs)

def build_env(
    env_id: str,
    env_spec: dict = {},
    env_kwargs: dict = {},
    env_wrappers: list = [],
    n_envs: int = 1,
    vectorization_mode: str = "auto",
    seed: int = None,
    obs_type: str = None,
    render_mode: str = None,
    grayscale_obs: bool = False,
    resize_obs: tuple = None,
    frame_stack: int = None,
    max_episode_steps: int = None,
    project_id: str = None,
    normalize_obs: bool = False,
    record_video: bool = False,
    record_video_kwargs: dict = {}
):
    import gymnasium as gym
    from gymnasium.wrappers.vector import RecordEpisodeStatistics
    from gym_wrappers.vec_env_info import VecEnvInfoWrapper
    from gym_wrappers.vec_normalize_static import VecNormalizeStatic
    from gym_wrappers.vec_video_recorder import VecVideoRecorder
   
    # Assert valid function arguments
    assert seed is not None, "Seed is required"
    assert frame_stack is None or frame_stack > 1, f"frame stack must be at least 2: frame_stack={frame_stack}"
    assert resize_obs is None or isinstance(resize_obs, (tuple, list)), f"resize obs must be a tuple or list: resize_obs={resize_obs}"
    assert resize_obs is None or len(resize_obs) == 2, f"resize obs must be a sequence of length 2: resize_obs={resize_obs}"
    assert resize_obs is None or all(x > 0 for x in resize_obs), f"resize obs must be positive: resize_obs={resize_obs}"

    # Convert resize_obs to tuple if it's a list (JSON deserialization converts tuples to lists)
    if resize_obs is not None and isinstance(resize_obs, list):
        resize_obs = tuple(resize_obs)
    assert not (record_video and obs_type != "rgb"), f"video recording requires rgb observations: obs_type={obs_type}"
    assert not (record_video and render_mode != "rgb_array"), f"video recording requires render_mode='rgb_array': render_mode={render_mode}"
    assert not (record_video and vectorization_mode == "async"), f"async vectorization does not support video recording: vectorization_mode={vectorization_mode}"

    # In case this is an ALE env, ensure envs are registered
    is_alepy_env = _is_alepy_env_id(env_id)
    if is_alepy_env: import ale_py; gym.register_envs(ale_py)
    
    # In case vectorization_mode is auto, resolve to 
    # atari native vectorization for ALE RGB envs
    is_ale_rgb_env = is_alepy_env and obs_type == "rgb"
    if vectorization_mode == "auto" and is_ale_rgb_env: vectorization_mode = "alepy" # TODO: use enum instead

    # In case vectorization_mode is auto, and this is a stable-retro env,
    # use async vectorization (multiprocessing) for Retro envs when n_envs > 1
    # (only one emulator core supported per process)
    is_stable_retro_env = _is_stable_retro_env_id(env_id)
    if vectorization_mode == "auto" and is_stable_retro_env: vectorization_mode = "async" if n_envs > 1 else "sync"

    # Create the vectorized environment
    if vectorization_mode == "alepy":
        vec_env = _build_vec_env_alepy(
            env_id,
            env_spec,
            env_kwargs,
            env_wrappers,
            n_envs,
            vectorization_mode,
            seed,
            obs_type,
            render_mode,
            grayscale_obs,
            resize_obs,
            frame_stack,
            record_video,
            record_video_kwargs,
            project_id,
        )
    else:
        vec_env = _build_vec_env_gym(
            env_id,
            env_spec,
            env_kwargs,
            env_wrappers,
            n_envs,
            vectorization_mode,
            seed,
            obs_type,
            render_mode,
            grayscale_obs,
            resize_obs,
            frame_stack,
            record_video,
            record_video_kwargs,
            max_episode_steps,
        )
    
    # Add episode statistics recorder wrapper
    vec_env = RecordEpisodeStatistics(vec_env)

    # Add static normalization wrapper (if requested)
    if normalize_obs == "static": vec_env = VecNormalizeStatic(vec_env)

    # Add video recorder wrapper (if requested)
    if record_video: vec_env = VecVideoRecorder(vec_env)

    # Add info wrapper (allows querying for env info)
    vec_env = VecEnvInfoWrapper(vec_env)

    # TODO: why is all this stuff needed?
    #vec_env._ale_atari_vec = True
    #vec_env.render_mode = render_mode
    #vec_env.env_id = env_id
    #vec_env._spec = spec
    #vec_env._project_id = project_id
    #vec_env._obs_type = obs_type
    #vec_env.render_mode = render_mode
    #vec_env.env_id = env_id

    # Return the vectorized environment
    return vec_env

def _build_vec_env_alepy(
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
    record_video: bool,
    record_video_kwargs: dict,
    project_id: str,
):
    from gymnasium import make_vec
    from gym_wrappers.ale_vec_video_recorder import ALEVecVideoRecorder

    assert obs_type == "rgb", "ALE native vectorization requires RGB observations"

    # Build kwargs for AtariVectorEnv
    # AtariVectorEnv supports grayscale, resizing, and frame stacking directly
    atari_kwargs = {}
    if grayscale_obs is not None:
        atari_kwargs['grayscale'] = grayscale_obs
    if resize_obs is not None:
        atari_kwargs['img_height'] = resize_obs[0]
        atari_kwargs['img_width'] = resize_obs[1]
    if frame_stack is not None:
        atari_kwargs['stack_num'] = frame_stack

    # Pass through relevant env_kwargs
    # TODO: map more env_kwargs to AtariVectorEnv parameters
    if 'repeat_action_probability' in env_kwargs:
        atari_kwargs['repeat_action_probability'] = env_kwargs['repeat_action_probability']
    if 'full_action_space' in env_kwargs:
        atari_kwargs['full_action_space'] = env_kwargs['full_action_space']

    vec_env = make_vec(
        env_id,
        num_envs=n_envs,
        vectorization_mode=None,
        **atari_kwargs,
    )

    # Seed the envs
    obs, _ = vec_env.reset(seed=seed)

    # Annotate env so VecEnvInfoWrapper can expose metadata/compat fallbacks
    vec_env._ale_atari_vec = True  # type: ignore[attr-defined]
    vec_env._spec = env_spec  # type: ignore[attr-defined]
    vec_env.env_id = env_id  # type: ignore[attr-defined]
    vec_env._project_id = project_id  # type: ignore[attr-defined]
    vec_env._obs_type = obs_type  # type: ignore[attr-defined]
    vec_env.render_mode = render_mode  # type: ignore[attr-defined]
    vec_env._last_seed = seed  # type: ignore[attr-defined]

    # TODO: is this working? how? is it slow?
    if record_video: vec_env = ALEVecVideoRecorder(vec_env, **record_video_kwargs)

    # Return the vectorized environment
    return vec_env

def _build_vec_env_gym(
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
    record_video: bool,
    record_video_kwargs: dict,
    max_episode_steps: int,
):
    from gym_wrappers.env_info import EnvInfoWrapper
    from gym_wrappers.env_video_recorder import EnvVideoRecorder

    # Resolve env type
    is_alepy_env = _is_alepy_env_id(env_id)
    is_vizdoom_env = _is_vizdoom_env_id(env_id)
    is_stable_retro_env = _is_stable_retro_env_id(env_id)
    is_bandit_env = _is_mab_env_id(env_id)

    def env_fn():
        from gymnasium.wrappers import (
            AtariPreprocessing,
            FrameStackObservation,
            GrayscaleObservation,
            ResizeObservation,
            TimeLimit,
        )

        local_env_kwargs = dict(env_kwargs)
        atari_frame_skip = None
        if is_alepy_env and obs_type == "rgb":
            configured_frameskip = local_env_kwargs.get("frameskip")
            atari_frame_skip = configured_frameskip if configured_frameskip is not None else 4
            base_frameskip = 1 if atari_frame_skip > 1 else atari_frame_skip
            local_env_kwargs["frameskip"] = base_frameskip

        if is_alepy_env: env = _build_env_alepy(env_id, obs_type, render_mode, **local_env_kwargs)
        elif is_vizdoom_env: env = _build_env_vizdoom(env_id, obs_type, render_mode, **local_env_kwargs)
        elif is_stable_retro_env: env = _build_env_stable_retro(env_id, obs_type, render_mode, **local_env_kwargs)
        elif is_bandit_env: env = _build_env_mab(env_id, obs_type, render_mode, **local_env_kwargs)
        else: env = _build_env_gym(env_id, obs_type, render_mode, **local_env_kwargs)

        if is_alepy_env and obs_type == "rgb":
            grayscale_flag = True if grayscale_obs is None else grayscale_obs
            screen_size = resize_obs if resize_obs is not None else 84
            if isinstance(screen_size, (tuple, list)):
                assert len(screen_size) == 2, f"resize_obs must have length 2: resize_obs={screen_size}"
                screen_size = (screen_size[1], screen_size[0])
            env = AtariPreprocessing(
                env,
                frame_skip=atari_frame_skip or 4,
                screen_size=screen_size,
                grayscale_obs=grayscale_flag,
                grayscale_newaxis=False,
                scale_obs=False,
            )

            # Apply custom wrappers on the preprocessed single-frame observations before stacking
            for wrapper in env_wrappers: env = EnvWrapperRegistry.apply(env, wrapper)

            if frame_stack: env = FrameStackObservation(env, stack_size=frame_stack, padding_type="zero")
        else:
            if grayscale_obs: env = GrayscaleObservation(env, keep_dim=False)
            if resize_obs: env = ResizeObservation(env, shape=resize_obs)

            # Apply custom wrappers before stacking so they operate on per-frame observations
            for wrapper in env_wrappers: env = EnvWrapperRegistry.apply(env, wrapper)

            if frame_stack: env = FrameStackObservation(env, stack_size=frame_stack)

        # Apply TimeLimit wrapper if max_episode_steps is specified
        if max_episode_steps is not None: env = TimeLimit(env, max_episode_steps=max_episode_steps)

        env = EnvInfoWrapper(env, obs_type=obs_type, project_id=env_id, spec=env_spec) # TODO: project id shouldn't be in env info wrapper

        if record_video: env = EnvVideoRecorder(env, **record_video_kwargs)

        return env

    # Create the vectorized environment
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    use_async = vectorization_mode == "async"
    vec_env_cls = AsyncVectorEnv if use_async else SyncVectorEnv
    vector_kwargs = {"context": "spawn"} if use_async else {} # TODO: is spawn still valid?
    env_fns = [lambda i=i: (e := env_fn(), e.reset(seed=seed + i), e)[2] for i in range(n_envs)]
    vec_env = vec_env_cls(env_fns, **vector_kwargs)

    # Return the vectorized environment
    return vec_env


def build_env_from_config(config, **kwargs):
    env_args = config.get_env_args()
    env_args.update(kwargs)
    env_id = env_args.pop("env_id")
    return build_env(env_id, **env_args)
