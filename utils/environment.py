from gym_wrappers.env_wrapper_registry import EnvWrapperRegistry

def is_alepy_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("ale/")

def is_vizdoom_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("vizdoom-")

def is_stable_retro_env_id(env_id: str) -> bool:
    return env_id.lower().startswith("retro/")

def is_rgb_env(env):
    import numpy as np
    from gymnasium import spaces

    # In case the observation space is not a box then it's not RGB
    if not isinstance(env.observation_space, spaces.Box):
        return False

    # If the observation space is not a 3D array then it's not RGB
    if len(env.observation_space.shape) < 3:
        return False

    # If the observation space is not uint8 then it's not RGB
    is_uint8 = getattr(env.observation_space, 'dtype', None) == np.uint8
    if not is_uint8:
        return False
    
    # If the observation space is not 3 channels then it's not RGB
    n_channels = env.observation_space.shape[-1]
    if not n_channels == 3:
        return False
    
    # Return True if all checks passed (is RGB)
    return True

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

def _build_env_gymnasium(env_id, obs_type, render_mode, **env_kwargs):
    import gymnasium as gym
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    return env  
    
def build_env(
    env_id, 
    n_envs=1, 
    seed=None, 
    max_episode_steps=None,
    env_wrappers=[], 
    grayscale_obs=False,
    resize_obs=False,
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
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
        VecNormalize
    )
    from gym_wrappers.vec_info import VecInfoWrapper
    from gym_wrappers.vec_normalize_static import VecNormalizeStatic
    from gym_wrappers.vec_video_recorder import VecVideoRecorder
        
    # If recording video was requrested, assert valid render mode and subproc disabled 
    if record_video and render_mode != "rgb_array": raise ValueError("Video recording requires render_mode='rgb_array'")
    if record_video and subproc: raise ValueError("Subprocess vector environments do not support video recording yet")

    # Determine the environment type
    _is_alepy_env = is_alepy_env_id(env_id)
    _is_vizdoom_env = is_vizdoom_env_id(env_id)
    _is_stable_retro_env = is_stable_retro_env_id(env_id)

    def env_fn():
        # Build the environment using the appropriate factory 
        if _is_alepy_env: env = _build_env_alepy(env_id, obs_type, render_mode, **env_kwargs)
        elif _is_vizdoom_env: env = _build_env_vizdoom(env_id, obs_type, render_mode, **env_kwargs)
        elif _is_stable_retro_env: env = _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs)            
        else: env = _build_env_gymnasium(env_id, obs_type, render_mode, **env_kwargs)

        # Apply configured env wrappers
        for wrapper in env_wrappers:
            env = EnvWrapperRegistry.apply(env, wrapper) # type: ignore

        # Truncate episode lengths if requested
        if max_episode_steps is not None:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Return the environment
        return env

    # Create the vectorized environment
    vec_env_cls = SubprocVecEnv if subproc else DummyVecEnv
    vec_env_kwargs = {"start_method": "spawn"} if vec_env_cls == SubprocVecEnv else {}
    env = make_vec_env(
        env_fn, 
        n_envs=n_envs, 
        seed=seed, 
        vec_env_cls=vec_env_cls, 
        vec_env_kwargs=vec_env_kwargs
    )

    # Ensure the vectorized env exposes render_mode for downstream wrappers (e.g., video recorder)
    setattr(env, "render_mode", render_mode)

    # Expose original env_id on the vectorized env for downstream introspection (e.g., YAML fallbacks)
    setattr(env, "env_id", env_id)
    
    # Detect image observations and transpose to channel-first for SB3 CNN policies
    from stable_baselines3.common.preprocessing import is_image_space
    is_image = is_image_space(env.observation_space, check_channels=False)

    from stable_baselines3.common.vec_env import VecTransposeImage
    # Only transpose if observations are channel-last images (H, W, C). Grayscale
    # Atari after preprocessing is (H, W) and should not be transposed here.
    obs_shape = getattr(env.observation_space, "shape", tuple())
    if is_image and isinstance(obs_shape, tuple) and len(obs_shape) == 3:
        env = VecTransposeImage(env)  # (N, C, H, W)

    # Enable observation normalization only for non-image observations
    if norm_obs == "static": env = VecNormalizeStatic(env)
    elif norm_obs == "rolling": env = VecNormalize(env, norm_obs=norm_obs)

    # Enable frame stacking if requested
    if frame_stack and frame_stack > 1: env = VecFrameStack(env, n_stack=frame_stack)
    
    # Enable video recording if requested
    # record_video_kwargs may include: video_length, record_env_idx (to record a single env)
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
