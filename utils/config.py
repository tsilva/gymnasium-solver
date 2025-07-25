"""Configuration management utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, MISSING
from typing import Union, Tuple, Dict, Any
from utils.misc import _convert_numeric_strings

@dataclass
class Config:
    # Environment
    env_id: str  # Environment ID string (e.g., 'CartPole-v1')
    algo_id: str  # Algorithm ID string (e.g., 'ppo', 'dqn')

    # Training (required fields first)
    n_steps: int
    batch_size: int
    n_epochs: int
    # Optional fields with defaults
    seed: int = 42  # Default: 42
    n_envs: int = 1  # Number of parallel environments (default: 1)

    # Networks
    hidden_dims: Union[int, Tuple[int, ...]] = (64,)  # Default: [64]
    # TODO: default learning rates should be in algo classes
    policy_lr: float = 0.0003  # Default: 0.0003
    #value_lr: float = 0.001  # Default: 0.001
    ent_coef: float = 0.01  # Default: 0.01
    val_coef: float = 0.5  # Default: 0.5 (for PPO)

    # Training
    max_epochs: int = None  # Default: -1
    gamma: float = 0.99  # Default: 0.99
    gae_lambda: float = 0.95  # Default: 0.95
    clip_range: float = 0.2  # Default: 0.2

    # Evaluation
    eval_rollout_interval: int = None  # Default: 10
    # TODO: this should be early_stop_on_reward_threshold, threshold should be on env spec
    eval_rollout_episodes: int = None
    eval_rollout_steps: int = None
    eval_async: bool = False  # Default: true (async evaluation)


    # Normalization
    normalize_obs: bool = False  # Default: false
    normalize_reward: bool = False  # Default: false
    
    # Reward Shaping (for environments like MountainCar)
    reward_shaping: Union[bool, Dict[str, Any]] = False  # Default: false

    @classmethod
    def load_from_yaml(cls, env_id: str, algo_id: str, config_dir: str = "configs") -> 'Config':
        """
        Load configuration from YAML files with hierarchical overrides:
        1. Start with RLConfig class defaults
        2. Apply environment-specific config (env_id.yaml -> default section)
        3. Apply algorithm-specific config (env_id.yaml -> algorithm section)
        """
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_dir

        # Start with class defaults
        final_config = {}
        for field in cls.__dataclass_fields__.values():
            if field.default is not MISSING:
                final_config[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore
                final_config[field.name] = field.default_factory()      # type: ignore

        # Load environment-specific configuration
        env_config_path = config_path / f"{env_id}.yaml"
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)

        # Set env_id and algo_id
        final_config['env_id'] = env_id
        final_config['algo_id'] = algo_id

        # Apply environment default config
        if 'default' in env_config:
            final_config.update(env_config['default'])

        # Apply algorithm-specific config
        algo_id_lower = algo_id.lower()
        if algo_id_lower in env_config:
            final_config.update(env_config[algo_id_lower])

        # Convert any numeric strings (like scientific notation)
        final_config = _convert_numeric_strings(final_config)

        # Convert list values to tuples for hidden_dims
        if 'hidden_dims' in final_config and isinstance(final_config['hidden_dims'], list):
            final_config['hidden_dims'] = tuple(final_config['hidden_dims'])

        instance = cls(**final_config)
        instance.validate()
        return instance
        
    def rollout_collector_hyperparams(self) -> Dict[str, Any]:
        return {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda
        }
    
    def validate(self) -> None:
        """Validate that all configuration values are valid."""
        # Environment
        if not self.env_id:
            raise ValueError("env_id must be a non-empty string.")
        if not self.algo_id:
            raise ValueError("algo_id must be a non-empty string.")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        # Networks
        if isinstance(self.hidden_dims, tuple) and not all(isinstance(x, int) for x in self.hidden_dims):
            raise ValueError("All elements of hidden_dims tuple must be ints.")
        if self.policy_lr <= 0:
            raise ValueError("policy_lr must be a positive float.")
        if self.ent_coef < 0:
            raise ValueError("ent_coef must be a non-negative float.")

        # Training
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1].")
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in [0, 1].")
        if not (0 < self.clip_range < 1):
            raise ValueError("clip_range must be in (0, 1).")

        # Evaluation
        if self.eval_rollout_interval is not None and self.eval_rollout_interval <= 0:
            raise ValueError("eval_rollout_interval must be a positive integer.")
        if self.eval_rollout_episodes is not None and self.eval_rollout_episodes <= 0:
            raise ValueError("eval_rollout_episodes must be a positive integer.")

def load_config(env_id: str, algo_id: str, config_dir: str = "configs") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(env_id, algo_id, config_dir)
