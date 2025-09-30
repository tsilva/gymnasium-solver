from gymnasium import logger as gym_logger

from gym_wrappers.env_wrapper_registry import EnvWrapperRegistry


def is_alepy_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("ale/")

def is_vizdoom_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("vizdoom-")

def is_stable_retro_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("retro/")

def is_mab_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("bandit-") or env_id.lower().startswith("bandit/") or env_id.lower() == "bandit-v0"

def _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs):
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    
    # In case the observation type is objects, use OCAtari 
    # to extract object-based observations from game RAM
    if obs_type == "objects": return _build_env_alepy__objects(env_id, obs_type, render_mode, **env_kwargs)
    elif obs_type == "rgb": return _build_env_alepy__rgb(env_id, obs_type, render_mode, **env_kwargs)
    elif obs_type == "ram": return _build_env_alepy__ram(env_id, obs_type, render_mode, **env_kwargs)
        
def _build_env_alepy__objects(env_id, obs_type, render_mode, **env_kwargs):
    from ocatari.core import OCAtari
    env = OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)
    return env

def _build_env_alepy__rgb(env_id, obs_type, render_mode, **env_kwargs):
    import gymnasium as gym

    # Create the environment
    env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)

    # Apply standard Atari preprocessing
    #from gymnasium.wrappers.atari import AtariPreprocessing  # type: ignore
    #env = AtariPreprocessing(
    #    env,
    #    grayscale_obs=True,
    #    scale_obs=False,
    #    screen_size=84,
    #    frame_skip=4,  #TODO: how to softcode these
    #    terminal_on_life_loss=False,
    #)

    # Return the environment
    return env

def _build_env_alepy__ram(env_id, obs_type, render_mode, **env_kwargs):
    import gymnasium as gym
    env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)
    return env

def _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs):
    import re

    from gym_wrappers.vizdoom import VizDoomEnv
    scenario = env_id.replace("VizDoom-", "").replace("-v0", "").replace("-v1", "").replace("-", "_")
    scenario = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', scenario)
    scenario = scenario.lower()
    env = VizDoomEnv(scenario=scenario, render_mode=render_mode, **env_kwargs)
    return env

def _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs):
    import retro  # type: ignore
    
    game = env_id.replace("Retro/", "")

    # Extract Retro-specific kwargs while keeping user overrides
    make_kwargs = dict(env_kwargs) if isinstance(env_kwargs, dict) else {}

    # Prefer a discrete action space so our categorical policies output
    # integer actions compatible with Retro's internal encoding.
    # Users can override via env_kwargs if they need MultiDiscrete/MultiBinary.
    make_kwargs.setdefault("use_restricted_actions", getattr(retro, "Actions").DISCRETE)

    # Support 'state' override via env_kwargs; None → retro's default state
    state = make_kwargs.pop("state", None)

    # stable-retro supports Gymnasium-style render_mode
    env = retro.make(
        game=game, 
        state=state, 
        render_mode=render_mode, 
        **make_kwargs
    )

    return env

def _build_env_mab(env_id, obs_type, render_mode, **env_kwargs):
    from gym_envs.mab_env import MultiArmedBanditEnv
    env = MultiArmedBanditEnv(**env_kwargs)
    return env

def _build_env_gym(env_id, obs_type, render_mode, **env_kwargs):
    import gymnasium as gym
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    return env  

# TODO: CLEAN this up

def build_env(
    env_id,
    n_envs=1,
    seed=None,
    max_episode_steps=None,
    project_id=None,
    spec=None,
    env_wrappers=[],
    grayscale_obs=False,
    resize_obs=False,
    normalize_obs=False,
    normalize_reward: bool = False,
    frame_stack=None,
    obs_type=None,
    render_mode=None,
    subproc=None,
    record_video=False,
    record_video_kwargs={},
    env_kwargs={}
):
    import gymnasium as gym
    from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics, GrayscaleObservation, ResizeObservation
    from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward

    from gym_wrappers.env_info import EnvInfoWrapper
    from gym_wrappers.env_video_recorder import EnvVideoRecorder
    from gym_wrappers.vec_env_info import VecEnvInfoWrapper
    from gym_wrappers.vec_normalize_static import VecNormalizeStatic
    from gym_wrappers.vec_video_recorder import VecVideoRecorder
        
    # If recording video was requrested, assert valid render mode and subproc disabled 
    if record_video and render_mode != "rgb_array": raise ValueError("Video recording requires render_mode='rgb_array'")
    if record_video and subproc: raise ValueError("Subprocess vector environments do not support video recording yet")

    # Determine the environment type
    _is_alepy_env = is_alepy_env_id(env_id)
    _is_vizdoom_env = is_vizdoom_env_id(env_id)
    _is_stable_retro_env = is_stable_retro_env_id(env_id)
    _is_bandit_env = is_mab_env_id(env_id)

    # ALE native vectorization for RGB environments (10x faster than standard vectorization)
    # Always use native vectorization for ALE RGB (includes grayscale, resize, frame_stack=4 by default)
    # Only supported for rgb obs_type; other obs_types (ram, objects) use standard vectorization
    _use_ale_native_vectorization = (
        _is_alepy_env
        and obs_type == "rgb"
        and n_envs > 1
    )

    # When using ALE RGB with n_envs=1, we need to match the native vectorization behavior
    # by forcing frame_stack=4 (native vectorization always uses 4 stacked frames)
    _is_ale_rgb_single_env = (
        _is_alepy_env
        and obs_type == "rgb"
        and n_envs == 1
    )
    if _is_ale_rgb_single_env and (frame_stack is None or frame_stack == 1):
        frame_stack = 4

    # Video recording not supported with ALE native vectorization
    if _use_ale_native_vectorization and record_video:
        record_video = False
        gym_logger.warn("Video recording not supported with ALE native vectorization; disabling video recording")

    def env_fn():
        # Build the environment using the appropriate factory
        if _is_alepy_env: env = _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs)
        elif _is_vizdoom_env: env = _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs)
        elif _is_stable_retro_env: env = _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs)
        elif _is_bandit_env: env = _build_env_mab(env_id, obs_type, render_mode, **env_kwargs)
        else: env = _build_env_gym(env_id, obs_type, render_mode, **env_kwargs)

        # Apply configured env wrappers
        for wrapper in env_wrappers:
            env = EnvWrapperRegistry.apply(env, wrapper) # type: ignore

        # Apply preprocessing wrappers for non-ALE envs if requested
        # (ALE native vectorization handles grayscale, resize, frame_stack automatically)
        if not _use_ale_native_vectorization:
            if grayscale_obs:
                # For ALE RGB single envs, use keep_dim=False to match native vectorization's CHW format
                # (native vec produces (4, 84, 84) after frame stacking, not (84, 84, 4))
                keep_dim = not _is_ale_rgb_single_env
                env = GrayscaleObservation(env, keep_dim=keep_dim)
            if resize_obs:
                # resize_obs can be True (default to 84x84) or [height, width]
                if resize_obs is True:
                    shape = (84, 84)
                elif isinstance(resize_obs, (list, tuple)) and len(resize_obs) == 2:
                    shape = tuple(resize_obs)
                else:
                    raise ValueError(f"resize_obs must be True or [height, width], got {resize_obs}")
                env = ResizeObservation(env, shape=shape)

        # TODO: modify existing timelimit if available
        # Truncate episode lengths if requested
        #if max_episode_steps is not None:
        #    from gymnasium.wrappers import TimeLimit
        #    env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Apply frame stacking at base env level if requested (no vector version exists)
        # Note: ALE native vectorization handles frame_stack=4 automatically
        if frame_stack and frame_stack > 1:
            env = FrameStackObservation(env, stack_size=frame_stack)

        # Record episode statistics (needed for rollout collector to track episode metrics)
        env = RecordEpisodeStatistics(env)

        # Attach metadata wrapper with context (obs_type and project_id)
        env = EnvInfoWrapper(env, obs_type=obs_type, project_id=project_id, spec=spec)

        # Enable video recording if requested
        if record_video: env = EnvVideoRecorder(env, **record_video_kwargs)

        # Return the environment
        return env

    # Create the vectorized environment
    if _use_ale_native_vectorization:
        # Use gymnasium.make_vec for ALE native vectorization (10x faster for RGB Atari)
        # When vectorization_mode is not specified, make_vec automatically uses ALE's native vectorizer
        from gymnasium import make_vec
        from gymnasium.wrappers.vector import RecordEpisodeStatistics as VectorRecordEpisodeStatistics
        import gymnasium as gym
        import ale_py

        # Register ALE environments before calling make_vec
        gym.register_envs(ale_py)

        # ALE native vectorization provides grayscale, resize to 84x84, and frame_stack=4 by default
        # Just pass through any explicit env_kwargs
        ale_kwargs = dict(env_kwargs) if env_kwargs else {}

        # ALE native vectorization doesn't support per-env wrappers, so we create
        # the vectorized env first and then apply vector-level wrappers
        env = make_vec(
            env_id,
            num_envs=n_envs,
            vectorization_mode=None,  # Auto-select: uses ALE native for Atari RGB
            **ale_kwargs
        )

        # Apply seed after vectorization if provided
        if seed is not None:
            env.reset(seed=seed)

        # Set attributes before wrapping (some wrappers don't allow setting attributes)
        setattr(env, "render_mode", render_mode)
        setattr(env, "env_id", env_id)

        # Apply vector-level RecordEpisodeStatistics to track episode metrics
        env = VectorRecordEpisodeStatistics(env)

        # Store metadata as attributes on the vectorized env
        # (ALE native doesn't support per-env wrappers, so we store at vec level)
        setattr(env, "_ale_native_vec", True)
        setattr(env, "_spec", spec)
        setattr(env, "_project_id", project_id)
        setattr(env, "_obs_type", obs_type)

        # Note: ALE native vectorization doesn't support per-env wrappers (env_wrappers,
        # frame_stack, video recording). Users needing these features should use non-RGB
        # obs types (ram, objects) which use standard vectorization.

    else:
        # Standard vectorization for non-ALE envs or non-RGB obs types
        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        vec_env_cls = AsyncVectorEnv if subproc else SyncVectorEnv
        vector_kwargs = {"context": "spawn"} if subproc else {}
        env_fns = [env_fn for _ in range(n_envs)]

        # Apply seed offset per env if seed is provided
        if seed is not None:
            for i, fn in enumerate(env_fns):
                original_fn = fn
                def make_env_with_seed(seed_val=seed + i, orig_fn=original_fn):
                    e = orig_fn()
                    e.reset(seed=seed_val)
                    return e
                env_fns[i] = make_env_with_seed

        env = vec_env_cls(env_fns, **vector_kwargs)

        # Set attributes after vectorization (for standard path only; ALE native sets before wrapping)
        setattr(env, "render_mode", render_mode)
        setattr(env, "env_id", env_id)
    
    # Observation/Reward normalization (Gymnasium wrappers or custom static wrapper)
    # Accepts:
    # - normalize_obs: False (default), True (rolling), 'rolling' (rolling), 'static' (bounds-based)
    # - normalize_reward: False (default) or True (rolling)
    if isinstance(normalize_obs, str) and normalize_obs.lower() == "static":
        # Static normalization of observations; reward normalization is not supported in this mode
        env = VecNormalizeStatic(env)
    else:
        # Rolling (running mean/std) normalization via Gymnasium wrappers
        use_norm_obs = bool(normalize_obs)  # True if normalize_obs is True/'rolling'
        if isinstance(normalize_obs, str):
            use_norm_obs = normalize_obs.lower() == "rolling"
        if use_norm_obs:
            env = NormalizeObservation(env)
        if normalize_reward:
            env = NormalizeReward(env)

    # Enable video recording if requested (proxies
    # to the underlying env video recorder)
    if record_video: env = VecVideoRecorder(env)

    # Wrap with vec env info wrapper that proxies to 
    # the underlying env info wrapper (first env metadata)
    env = VecEnvInfoWrapper(env)

    return env

def build_env_from_config(config, **kwargs):
    env_args = config.get_env_args()
    env_args.update(kwargs)
    env_id = env_args.pop("env_id")
    return build_env(env_id, **env_args)
