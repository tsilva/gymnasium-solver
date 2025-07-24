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
    env_spec: Dict[str, Any] = None  # Environment specification dictionary
    seed: int = 42  # Random seed for reproducibility

    # Networks
    hidden_dims: Union[int, Tuple[int, ...]] = 64  # Hidden layer dimensions for neural networks
    policy_lr: float = 3e-4  # Learning rate for the policy network
    value_lr: float = 1e-3  # Learning rate for the value network
    entropy_coef: float = 0.01  # Entropy coefficient for exploration

    # Training
    max_epochs: int = -1  # Maximum number of training epochs (-1 for unlimited)
    train_rollout_interval: int = 10  # Interval (in epochs) between training rollouts
    train_rollout_steps: int = 2048  # Number of steps per training rollout
    train_reward_threshold: float = None  # Reward threshold to stop training early
    train_batch_size: int = 64  # Batch size for training updates
    gamma: float = 0.99  # Discount factor for future rewards
    gae_lambda: float = 0.95  # Lambda for Generalized Advantage Estimation (GAE)
    clip_epsilon: float = 0.2  # Clipping epsilon for PPO or similar algorithms

    # Evaluation
    eval_rollout_interval: int = 10  # Interval (in epochs) between evaluation rollouts
    eval_rollout_episodes: int = 32  # Number of episodes per evaluation rollout
    eval_reward_threshold: float = None  # Reward threshold for evaluation

    # Normalization
    normalize_obs: bool = False  # Whether to normalize observations
    normalize_reward: bool = False  # Whether to normalize rewards

    # Miscellaneous
    mean_reward_window: int = 100  # Window size for calculating mean reward
    
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
        
        return cls(**final_config)
    

    def rollout_collector_hyperparams(self) -> Dict[str, Any]:
        return {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda
        }


def load_config(env_id: str, algo_id: str, config_dir: str = "configs") -> RLConfig:
    """Convenience function to load configuration."""
    return RLConfig.load_from_yaml(env_id, algo_id, config_dir)
