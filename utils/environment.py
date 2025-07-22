"""Environment setup utilities."""

import multiprocessing

def _build_env(env_id, n_envs=1, seed=None, norm_obs=False, norm_reward=False):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    if n_envs == 'auto': n_envs = multiprocessing.cpu_count()
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env

def setup_environment(config, n_envs="auto"):
    """Setup environment with configuration."""
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    # Create environment builder
    def _build_env_fn(seed):
        return _build_env(
            config.env_id,
            norm_obs=config.normalize,
            n_envs=n_envs,
            seed=seed
        )
    return _build_env_fn
