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
    assert resize_obs is None or isinstance(resize_obs, tuple), f"resize obs must be a tuple: resize_obs={resize_obs}"
    assert resize_obs is None or len(resize_obs) == 2, f"resize obs must be a tuple of length 2: resize_obs={resize_obs}"
    assert resize_obs is None or all(x > 0 for x in resize_obs), f"resize obs must be positive: resize_obs={resize_obs}"
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
    _build_vec_env_fn = vectorization_mode == "alepy" and _build_vec_env_alepy or _build_vec_env_gym
    vec_env = _build_vec_env_fn(
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
):
    from gymnasium import make_vec
    from gym_wrappers.ale_vec_video_recorder import ALEVecVideoRecorder

    assert obs_type == "rgb", "ALE native vectorization requires RGB observations"

    # TODO: pass all frame_stack, grayscale_obs, resize_obs, frameskip, etc to make_vec
    vec_env = make_vec(
        env_id, 
        num_envs=n_envs, 
        vectorization_mode=None, 
        #**env_kwargs
    )

    # Seed the envs
    obs, _ = vec_env.reset(seed=seed)
    
    # Assert that the obs shape is as expected 
    obs_shape = obs.shape[1:]
    expected_shape = (frame_stack, *resize_obs)
    assert obs_shape == expected_shape, f"ALE native vectorization expected {expected_shape} but got {obs_shape}."

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
):
    from gym_wrappers.env_info import EnvInfoWrapper
    from gym_wrappers.env_video_recorder import EnvVideoRecorder

    # Resolve env type
    is_alepy_env = _is_alepy_env_id(env_id)
    is_vizdoom_env = _is_vizdoom_env_id(env_id)
    is_stable_retro_env = _is_stable_retro_env_id(env_id)
    is_bandit_env = _is_mab_env_id(env_id)

    def env_fn():
        if is_alepy_env: env = _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs)
        elif is_vizdoom_env: env = _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs)
        elif is_stable_retro_env: env = _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs)
        elif is_bandit_env: env = _build_env_mab(env_id, obs_type, render_mode, **env_kwargs)
        else: env = _build_env_gym(env_id, obs_type, render_mode, **env_kwargs)

        from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
        if grayscale_obs: env = GrayscaleObservation(env, keep_dim=False)
        if resize_obs: env = ResizeObservation(env, shape=resize_obs)
        if frame_stack: env = FrameStackObservation(env, stack_size=frame_stack)

        # Apply custom wrappers first (before frame stacking) so they operate on raw observations
        for wrapper in env_wrappers: env = EnvWrapperRegistry.apply(env, wrapper)

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
