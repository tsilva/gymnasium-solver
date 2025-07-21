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
    env_id: str
    seed: int = 42
    
    # Training
    max_epochs: int = -1
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    batch_size: int = 64
    train_rollout_steps: int = 2048
    
    # Evaluation
    eval_interval: int = 10
    eval_episodes: int = 32
    reward_threshold: float = 200
    
    # Networks
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    hidden_dims: Union[int, Tuple[int, ...]] = 64
    entropy_coef: float = 0.01
    
    # Other
    normalize: bool = False
    mean_reward_window: int = 100
    rollout_interval: int = 10
    
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
        
        return cls(**final_config)


def load_config(env_id: str, algo_id: str, config_dir: str = "configs") -> RLConfig:
    """Convenience function to load configuration."""
    return RLConfig.load_from_yaml(env_id, algo_id, config_dir)
