"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
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
    hidden_dim: Union[int, Tuple[int, ...]] = 64
    entropy_coef: float = 0.01
    
    # Shared backbone (PPO only)
    shared_backbone: bool = False
    backbone_dim: Union[int, Tuple[int, ...]] = 64
    
    # Other
    normalize: bool = False
    mean_reward_window: int = 100
    rollout_interval: int = 10
    n_envs: Union[str, int] = "auto"
    async_rollouts: bool = True
    
    @classmethod
    def load_from_yaml(cls, env_id: str, algorithm: str = "ppo", config_dir: str = "configs") -> 'RLConfig':
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
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                default_config = yaml.safe_load(f)
        else:
            default_config = {}
        
        # Load environment-specific configuration
        env_config_path = config_path / f"{env_id}.yaml"
        env_config = {}
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
        
        # Start with default config
        final_config = default_config.copy()
        final_config['env_id'] = env_id
        
        # Apply environment default config
        if 'default' in env_config:
            final_config.update(env_config['default'])
        
        # Apply algorithm-specific config
        algorithm_lower = algorithm.lower()
        if algorithm_lower in env_config:
            final_config.update(env_config[algorithm_lower])
        
        # Convert any numeric strings (like scientific notation)
        final_config = _convert_numeric_strings(final_config)
        
        # Convert list values to tuples for hidden_dim and backbone_dim
        if 'hidden_dim' in final_config and isinstance(final_config['hidden_dim'], list):
            final_config['hidden_dim'] = tuple(final_config['hidden_dim'])
        if 'backbone_dim' in final_config and isinstance(final_config['backbone_dim'], list):
            final_config['backbone_dim'] = tuple(final_config['backbone_dim'])
        
        return cls(**final_config)


def load_config(env_id: str, algorithm: str = "ppo", config_dir: str = "configs") -> RLConfig:
    """Convenience function to load configuration."""
    return RLConfig.load_from_yaml(env_id, algorithm, config_dir)
