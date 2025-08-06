from wrappers.env_wrapper_registry import EnvWrapperRegistry
from wrappers.discrete_to_onehot import DiscreteToOneHot
from gymnasium import spaces

def is_atari_env_id(env_id: str) -> bool:
    return env_id.startswith("ALE/")

def build_env(
    env_id, 
    n_envs=1, 
    seed=None, 
    env_wrappers=[], 
    norm_obs=False, 
    frame_stack=None, 
    obs_type=None,
    render_mode=None,
    subproc=None,
    record_video=False, 
    record_video_kwargs={},
    env_kwargs={}
):

    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv, SubprocVecEnv
    from wrappers.vec_info import VecInfoWrapper
    from wrappers.vec_video_recorder import VecVideoRecorder
    from wrappers.vec_normalize_static import VecNormalizeStatic
    
    # If recording video was requrested, assert valid render mode and subproc disabled 
    if record_video:
        if render_mode != "rgb_array": raise ValueError("Video recording requires render_mode='rgb_array'")
        if subproc: raise ValueError("Subprocess vector environments do not support video recording yet")

    _is_atari = is_atari_env_id(env_id)

    def env_fn():
        if _is_atari: 
            # Import and register ALE environments in each subprocess
            import ale_py
            gym.register_envs(ale_py)

            if obs_type == "objects":
                from ocatari.core import OCAtari
                env = OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode, **env_kwargs)
            else:
                env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode, **env_kwargs)
        else: 
            env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

        # Automatically apply DiscreteToOneHot wrapper for discrete observation spaces
        if isinstance(env.observation_space, spaces.Discrete):
            env = DiscreteToOneHot(env)

        # Apply configured env wrappers
        for wrapper in env_wrappers: env = EnvWrapperRegistry.apply(env, wrapper)

        # Return the environment
        return env

    # Vectorize the environment
    vec_env_cls = DummyVecEnv
    if subproc is not None: vec_env_cls = SubprocVecEnv if subproc else DummyVecEnv
    elif _is_atari: vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    vec_env_kwargs = {"start_method": "spawn"} if vec_env_cls == SubprocVecEnv else {}
    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)

    # Enable observation normalization if requested
    if norm_obs == "static": env = VecNormalizeStatic(env)
    elif norm_obs is True: env = VecNormalize(env, norm_obs=norm_obs)
    
    # Enable frame stacking if requested
    if frame_stack and frame_stack > 1: env = VecFrameStack(env, n_stack=frame_stack)
    
    # Enable video recording if requested
    if record_video:
        env = VecVideoRecorder(
            env,
            **record_video_kwargs
        )

    # Add instrospection info wrapper that 
    # allows easily querying for env details
    # through the vectorized wrapper
    # This should be added last to get the final observation space dimensions
    env = VecInfoWrapper(env)

    return env
