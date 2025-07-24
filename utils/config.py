"""Configuration management utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any


def _convert_numeric_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string representations of numbers back to numeric types."""
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            if 'e' in value.lower() or 'E' in value:
                try:
                    config_dict[key] = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
    return config_dict

@dataclass
class RLConfig:
    """Reinforcement Learning Configuration."""
    
    # Environment
    env_id: str  # Environment ID string (e.g., 'CartPole-v1')
    algo_id: str  # Algorithm ID string (e.g., 'ppo', 'dqn')
    env_spec: Dict[str, Any]  # Environment specification dictionary
    seed: int  # Random seed for reproducibility

    # Networks
    hidden_dims: Union[int, Tuple[int, ...]]  # Hidden layer dimensions for neural networks
    policy_lr: float  # Learning rate for the policy network
    value_lr: float  # Learning rate for the value network
    entropy_coef: float  # Entropy coefficient for exploration

    # Training
    max_epochs: int  # Maximum number of training epochs
    train_rollout_interval: int  # Interval (in epochs) between training rollouts
    train_rollout_steps: int  # Number of steps per training rollout
    train_reward_threshold: float  # Reward threshold to stop training early
    train_batch_size: int  # Batch size for training updates
    gamma: float  # Discount factor for future rewards
    gae_lambda: float  # Lambda for Generalized Advantage Estimation (GAE)
    clip_epsilon: float  # Clipping epsilon for PPO or similar algorithms

    # Evaluation
    eval_rollout_interval: int  # Interval (in epochs) between evaluation rollouts
    eval_rollout_episodes: int  # Number of episodes per evaluation rollout
    eval_reward_threshold: float  # Reward threshold for evaluation

    # Normalization
    normalize_obs: bool  # Whether to normalize observations
    normalize_reward: bool  # Whether to normalize rewards

    # Miscellaneous
    mean_reward_window: int  # Window size for calculating mean reward
    
    @classmethod
    def load_from_yaml(cls, env_id: str, algo_id: str, config_dir: str = "configs") -> 'RLConfig':
        """
        Load configuration from YAML files with hierarchical overrides:
        1. Start with default.yaml
        2. Apply environment-specific config (env_id.yaml -> default section)
        3. Apply algorithm-specific config (env_id.yaml -> algorithm section)
        """
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_dir
        
        # Load default configuration
        default_config_path = config_path / "default.yaml"
        with open(default_config_path, 'r') as f: default_config = yaml.safe_load(f)
        
        # Load environment-specific configuration
        env_config_path = config_path / f"{env_id}.yaml"
        with open(env_config_path, 'r') as f: env_config = yaml.safe_load(f)
        
        # Start with default config
        final_config = default_config.copy()
        final_config['env_id'] = env_id
        final_config['algo_id'] = algo_id
        
        # Apply environment default config
        if 'default' in env_config:
            final_config.update(env_config['default'])
        
        # Apply algorithm-specific config
        algo_id = algo_id.lower()
        if algo_id in env_config: final_config.update(env_config[algo_id])

        # Convert any numeric strings (like scientific notation)
        final_config = _convert_numeric_strings(final_config)
        
        # Convert list values to tuples for hidden_dims
        if 'hidden_dims' in final_config and isinstance(final_config['hidden_dims'], list):
            final_config['hidden_dims'] = tuple(final_config['hidden_dims'])
        
        # TODO: better way to do this?
        from utils.environment import build_env, get_env_spec
        env = build_env(env_id)
        env_spec = get_env_spec(env)
        final_config['env_spec'] = env_spec
        
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
        if self.env_spec is None:
            raise ValueError("env_spec must not be None.")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        # Networks
        if isinstance(self.hidden_dims, tuple) and not all(isinstance(x, int) for x in self.hidden_dims):
            raise ValueError("All elements of hidden_dims tuple must be ints.")
        if self.policy_lr <= 0:
            raise ValueError("policy_lr must be a positive float.")
        if self.value_lr <= 0:
            raise ValueError("value_lr must be a positive float.")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be a non-negative float.")

        # Training
        if self.train_rollout_interval <= 0:
            raise ValueError("train_rollout_interval must be a positive integer.")
        if self.train_rollout_steps <= 0:
            raise ValueError("train_rollout_steps must be a positive integer.")
        if self.train_reward_threshold is not None and not isinstance(self.train_reward_threshold, (float, int)):
            raise ValueError("train_reward_threshold must be None or a number.")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be a positive integer.")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1].")
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in [0, 1].")
        if not (0 < self.clip_epsilon < 1):
            raise ValueError("clip_epsilon must be in (0, 1).")

        # Evaluation
        if self.eval_rollout_interval <= 0:
            raise ValueError("eval_rollout_interval must be a positive integer.")
        if self.eval_rollout_episodes <= 0:
            raise ValueError("eval_rollout_episodes must be a positive integer.")
        if self.eval_reward_threshold is not None and not isinstance(self.eval_reward_threshold, (float, int)):
            raise ValueError("eval_reward_threshold must be None or a number.")

        # Miscellaneous
        if self.mean_reward_window <= 0:
            raise ValueError("mean_reward_window must be a positive integer.")

def load_config(env_id: str, algo_id: str, config_dir: str = "configs") -> RLConfig:
    """Convenience function to load configuration."""
    return RLConfig.load_from_yaml(env_id, algo_id, config_dir)
