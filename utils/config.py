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
    n_timesteps: float = None  # Total timesteps for training (RLZOO format)
    policy: str = 'MlpPolicy'  # Policy type (RLZOO format)

    # Networks
    hidden_dims: Union[int, Tuple[int, ...]] = (64,)  # Default: [64]
    # TODO: default learning rates should be in algo classes
    policy_lr: float = 0.0003  # Default: 0.0003
    learning_rate: float = None  # RLZOO format learning rate (overrides policy_lr if set)
    #value_lr: float = 0.001  # Default: 0.001
    ent_coef: float = 0.01  # Default: 0.01
    val_coef: float = 0.5  # Default: 0.5 (for PPO)
    vf_coef: float = None  # RLZOO format value function coefficient

    max_grad_norm: float = 0.5  # Default: 0.5 (for gradient clipping)
    
    # Training
    max_epochs: int = None  # Default: -1
    gamma: float = 0.99  # Default: 0.99
    gae_lambda: float = 0.95  # Default: 0.95
    clip_range: float = 0.2  # Default: 0.2

    # Additional RLZOO format parameters
    normalize: bool = None  # RLZOO format normalization flag
    use_sde: bool = False  # Use State Dependent Exploration
    sde_sample_freq: int = -1  # SDE sample frequency
    policy_kwargs: str = None  # Policy kwargs as string

    # Evaluation
    eval_freq_epochs: int = 10
    eval_episodes: int = 10
    eval_async: bool = False  # Default: true (async evaluation)
    eval_deterministic: bool = True  # Default: true (deterministic evaluation)


    # Normalization
    normalize_obs: bool = False  # Default: false
    normalize_reward: bool = False  # Default: false
    
    # Frame stacking
    frame_stack: int = 1  # Default: 1 (no frame stacking)
    
    # Atari-specific settings
    obs_type: str = "rgb"  # Default: "rgb" (other options: "ram", "grayscale")
    
    # TODO: generalize to just add custom wrappers
    # Reward Shaping (for environments like MountainCar)
    reward_shaping: Union[bool, Dict[str, Any]] = False  # Default: false

    @classmethod
    def load_from_yaml(cls, env_id: str, algo_id: str, config_dir: str = "hyperparams") -> 'Config':
        """
        Load configuration from YAML files in RLZOO SB3 format:
        1. Start with RLConfig class defaults
        2. Load algorithm-specific file (hyperparams/{algo_id}.yaml)
        3. Apply environment-specific config from that file
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

        # Load algorithm-specific configuration file
        algo_config_path = config_path / f"{algo_id.lower()}.yaml"
        if not algo_config_path.exists():
            raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")
        
        with open(algo_config_path, 'r') as f:
            algo_config = yaml.safe_load(f)

        # Set env_id and algo_id
        final_config['env_id'] = env_id
        final_config['algo_id'] = algo_id

        # Apply environment-specific config from algorithm file
        if env_id in algo_config:
            final_config.update(algo_config[env_id])
        else:
            raise ValueError(f"Environment '{env_id}' not found in {algo_config_path}")

        # Convert any numeric strings (like scientific notation)
        final_config = _convert_numeric_strings(final_config)

        # Handle RLZOO format compatibility
        # Use learning_rate if set, otherwise use policy_lr
        if 'learning_rate' in final_config and final_config['learning_rate'] is not None:
            final_config['policy_lr'] = final_config['learning_rate']
        
        # Handle normalize flag (RLZOO format) -> normalize_obs
        if 'normalize' in final_config and final_config['normalize'] is not None:
            final_config['normalize_obs'] = final_config['normalize']
            final_config['normalize_reward'] = final_config['normalize']
        
        # Handle vf_coef -> val_coef
        if 'vf_coef' in final_config and final_config['vf_coef'] is not None:
            final_config['val_coef'] = final_config['vf_coef']

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
        if self.eval_freq_epochs is not None and self.eval_freq_epochs <= 0:
            raise ValueError("eval_freq_epochs must be a positive integer.")
        if self.eval_episodes is not None and self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be a positive integer.")

def load_config(env_id: str, algo_id: str, config_dir: str = "hyperparams") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(env_id, algo_id, config_dir)
