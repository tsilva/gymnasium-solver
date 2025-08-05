try:
    import ale_py
    import gymnasium
    gymnasium.register_envs(ale_py)
except ImportError:
    # ale_py not available, skip Atari environment registration
    import gymnasium

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
    record_video=False, 
    record_video_kwargs={}
):

    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
    from wrappers.vec_info import VecInfoWrapper
    from wrappers.vec_video_recorder import VecVideoRecorder
    from wrappers.vec_normalize_static import VecNormalizeStatic
    
    # Assert render_mode is set if recording video
    if record_video and render_mode != "rgb_array":
        raise ValueError("Video recording requires render_mode='rgb_array'")

    def env_fn():
        # TODO: is there overlap here?
        if is_atari_env_id(env_id): 
            if obs_type == "objects":
                from ocatari.core import OCAtari
                env = OCAtari(env_id, mode="ram", hud=False, render_mode=render_mode)
            else:
                env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode)
        else: 
            env = gym.make(env_id, render_mode=render_mode)

        # Automatically apply DiscreteToOneHot wrapper for discrete observation spaces
        if isinstance(env.observation_space, spaces.Discrete):
            env = DiscreteToOneHot(env)

        # Apply configured env wrappers
        for wrapper in env_wrappers: env = EnvWrapperRegistry.apply(env, wrapper)

        # Return the environment
        return env

    # Vectorize the environment
    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)

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
