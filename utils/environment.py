"""Environment setup utilities."""

from typing import Callable
import multiprocessing

def set_random_seed(seed):
    from stable_baselines3.common.utils import set_random_seed as _set_random_seed
    _set_random_seed(seed)

def _build_env(env_id, n_envs=1, seed=None, norm_obs=False, norm_reward=False):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    if n_envs == 'auto': n_envs = multiprocessing.cpu_count()
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env

# TODO: get rid of this
def create_env_builder(config) -> Callable:
    """Create environment builder function with config parameters."""
    def build_env_fn(seed, n_envs=None):
        return _build_env(
            config.env_id,
            norm_obs=config.normalize,
            n_envs=n_envs if n_envs is not None else config.n_envs,
            seed=seed
        )
    return build_env_fn

def setup_environment(config):
    """Setup environment with configuration."""
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    
    # Create environment builder
    build_env_fn = create_env_builder(config)
    
    return build_env_fn
