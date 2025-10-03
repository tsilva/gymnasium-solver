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

    if obs_type == "objects":
        from ocatari.core import OCAtari
        return OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)

    assert obs_type in ("ram", "rgb"), f"Unsupported obs_type for ALE: {obs_type}"
    return gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)

def _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs):
    import re
    from gym_wrappers.vizdoom import VizDoomEnv

    scenario = env_id.replace("VizDoom-", "").replace("-v0", "").replace("-v1", "").replace("-", "_")
    scenario = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', scenario).lower()
    return VizDoomEnv(scenario=scenario, render_mode=render_mode, **env_kwargs)

def _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs):
    try:
        import retro  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"stable-retro is required for {env_id} but not installed. "
            f"Note: stable-retro 0.9.5 is broken on M1 Mac (arm64 wheel contains x86_64 binary). "
            f"Install with: uv pip install .[retro]"
        ) from e

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
    vectorization_mode="auto",
    record_video=False,
    record_video_kwargs={},
    env_kwargs={}
):
    import gymnasium as gym
    from gymnasium.wrappers import RecordEpisodeStatistics
    from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward

    from gym_wrappers.env_info import EnvInfoWrapper
    from gym_wrappers.env_video_recorder import EnvVideoRecorder
    from gym_wrappers.vec_env_info import VecEnvInfoWrapper
    from gym_wrappers.vec_normalize_static import VecNormalizeStatic
    from gym_wrappers.vec_video_recorder import VecVideoRecorder

    assert not (record_video and render_mode != "rgb_array"), "Video recording requires render_mode='rgb_array'"
    assert not (record_video and vectorization_mode == "async"), "Async vectorization does not support video recording"

    _is_alepy_env = is_alepy_env_id(env_id)
    _is_vizdoom_env = is_vizdoom_env_id(env_id)
    _is_stable_retro_env = is_stable_retro_env_id(env_id)
    _is_bandit_env = is_mab_env_id(env_id)

    # stable-retro doesn't support multiple emulator instances per process
    # Force async vectorization (multiprocessing) for Retro envs when n_envs > 1
    if _is_stable_retro_env and n_envs > 1 and vectorization_mode == "auto":
        vectorization_mode = "async"

    if _is_alepy_env:
        import ale_py
        gym.register_envs(ale_py)

    _is_ale_rgb_env = _is_alepy_env and obs_type == "rgb"
    _use_ale_atari_vectorization = _is_ale_rgb_env and vectorization_mode in ("auto", "atari")
    if _use_ale_atari_vectorization:
        from gymnasium import make_vec
        from gymnasium.wrappers.vector import RecordEpisodeStatistics as VectorRecordEpisodeStatistics
        from gym_wrappers.ale_vec_video_recorder import ALEVecVideoRecorder

        env = make_vec(env_id, num_envs=n_envs, vectorization_mode=None, **env_kwargs)
        if seed is not None:
            env.reset(seed=seed)

        obs_space = env.single_observation_space
        assert hasattr(obs_space, 'shape'), "ALE native vec env must have observation space shape"
        assert obs_space.shape == (4, 84, 84), (
            f"ALE native vectorization expected (4, 84, 84) but got {obs_space.shape}. "
            f"Config: frame_stack={frame_stack}, grayscale_obs={grayscale_obs}, resize_obs={resize_obs}"
        )

        env.render_mode = render_mode
        env.env_id = env_id
        env = VectorRecordEpisodeStatistics(env)

        if record_video:
            env = ALEVecVideoRecorder(env, **record_video_kwargs)

        env._ale_atari_vec = True
        env._spec = spec
        env._project_id = project_id
        env._obs_type = obs_type
    else:
        def env_fn():
            if _is_alepy_env:
                env = _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs)
            elif _is_vizdoom_env:
                env = _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs)
            elif _is_stable_retro_env:
                env = _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs)
            elif _is_bandit_env:
                env = _build_env_mab(env_id, obs_type, render_mode, **env_kwargs)
            else:
                env = _build_env_gym(env_id, obs_type, render_mode, **env_kwargs)

            from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, FlattenObservation

            # Apply custom wrappers first (before frame stacking) so they operate on raw observations
            for wrapper in env_wrappers:
                env = EnvWrapperRegistry.apply(env, wrapper)

            if _is_ale_rgb_env:
                env = GrayscaleObservation(env, keep_dim=False)
                env = ResizeObservation(env, shape=(84, 84))
                env = FrameStackObservation(env, stack_size=4)
            else:
                if grayscale_obs:
                    env = GrayscaleObservation(env, keep_dim=False)
                if resize_obs:
                    resize_shape = (84, 84) if resize_obs is True else tuple(resize_obs) if isinstance(resize_obs, list) else resize_obs
                    env = ResizeObservation(env, shape=resize_shape)
                if frame_stack and frame_stack > 1:
                    env = FrameStackObservation(env, stack_size=frame_stack)
                    # Flatten stacked vector observations for MLP compatibility
                    # Images (3D obs) stay multi-dimensional; vectors (1D->2D after stacking) get flattened
                    if len(env.observation_space.shape) == 2:
                        env = FlattenObservation(env)

            env = RecordEpisodeStatistics(env)
            env = EnvInfoWrapper(env, obs_type=obs_type, project_id=project_id, spec=spec)

            if record_video:
                env = EnvVideoRecorder(env, **record_video_kwargs)

            return env

        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        use_async = vectorization_mode == "async"
        vec_env_cls = AsyncVectorEnv if use_async else SyncVectorEnv
        vector_kwargs = {"context": "spawn"} if use_async else {}

        if seed is not None:
            env_fns = [lambda i=i: (e := env_fn(), e.reset(seed=seed + i), e)[2] for i in range(n_envs)]
        else:
            env_fns = [env_fn for _ in range(n_envs)]

        env = vec_env_cls(env_fns, **vector_kwargs)
        env.render_mode = render_mode
        env.env_id = env_id
    
    if isinstance(normalize_obs, str) and normalize_obs.lower() == "static":
        env = VecNormalizeStatic(env)
    else:
        if normalize_obs and (not isinstance(normalize_obs, str) or normalize_obs.lower() == "rolling"):
            env = NormalizeObservation(env)
        if normalize_reward:
            env = NormalizeReward(env)

    if record_video:
        env = VecVideoRecorder(env)

    env = VecEnvInfoWrapper(env)

    return env

def build_env_from_config(config, **kwargs):
    env_args = config.get_env_args()
    env_args.update(kwargs)
    env_id = env_args.pop("env_id")
    return build_env(env_id, **env_args)
