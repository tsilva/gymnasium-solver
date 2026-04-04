import gymnasium as gym
from utils.environment_builders import build_single_env
from utils.environment_types import get_env_type
from utils.environment_vectorizers import (
    build_vec_env_alepy,
    build_vec_env_gym,
    resolve_vectorization_mode,
)


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
    frame_skip: int = None,
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
    assert frame_stack is None or frame_stack >= 1, f"frame stack must be at least 1: frame_stack={frame_stack}"
    assert frame_skip is None or frame_skip >= 1, f"frame skip must be at least 1: frame_skip={frame_skip}"
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
    is_alepy_env = get_env_type(env_id) == 'alepy'
    if is_alepy_env: import ale_py; gym.register_envs(ale_py)
    
    # In case vectorization_mode is auto, resolve to 
    # atari native vectorization for ALE RGB envs
    is_ale_rgb_env = is_alepy_env and obs_type == "rgb"
    vectorization_mode = resolve_vectorization_mode(env_id, obs_type, vectorization_mode, n_envs)

    # Create the vectorized environment
    if vectorization_mode == "alepy":
        vec_env = build_vec_env_alepy(
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
            frame_skip,
            record_video,
            record_video_kwargs,
            project_id,
        )
    else:
        vec_env = build_vec_env_gym(
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
            frame_skip,
            record_video,
            record_video_kwargs,
            max_episode_steps,
        )
    
    # Add episode statistics recorder wrapper
    vec_env = RecordEpisodeStatistics(vec_env)

    # Add observation normalization wrapper (if requested)
    if normalize_obs == "static":
        vec_env = VecNormalizeStatic(vec_env)
    elif normalize_obs in (True, "rolling"):
        from gymnasium.wrappers.vector import NormalizeObservation
        vec_env = NormalizeObservation(vec_env)

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


def build_env_from_config(config, **kwargs):
    env_args = config.get_env_args()
    env_args.update(kwargs)
    env_id = env_args.pop("env_id")
    return build_env(env_id, **env_args)
