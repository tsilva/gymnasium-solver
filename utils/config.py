"""Configuration management utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, MISSING
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

    # Training (required fields first)
    n_steps: int
    batch_size: int
    # Optional fields with defaults
    seed: int = 42  # Default: 42
    n_envs: int = 1  # Number of parallel environments (default: 1)

    # Networks
    hidden_dims: Union[int, Tuple[int, ...]] = (64,)  # Default: [64]
    policy_lr: float = 0.0003  # Default: 0.0003
    value_lr: float = 0.001  # Default: 0.001
    ent_coef: float = 0.01  # Default: 0.01

    # Training
    # TODO: make max_epochas = None
    max_epochs: int = -1  # Default: -1
    train_rollout_interval: int = 1  # Default: 1
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

    # Miscellaneous
    mean_reward_window: int = 100  # Default: 100
    
    @classmethod
    def load_from_yaml(cls, env_id: str, algo_id: str, config_dir: str = "configs") -> 'RLConfig':
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
        if self.ent_coef < 0:
            raise ValueError("ent_coef must be a non-negative float.")

        # Training
        if self.train_rollout_interval <= 0:
            raise ValueError("train_rollout_interval must be a positive integer.")
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

        # Miscellaneous
        if self.mean_reward_window <= 0:
            raise ValueError("mean_reward_window must be a positive integer.")

    def __str__(self) -> str:
        """Return a human-readable string representation of the configuration."""
        lines = ["Configuration", "=" * 40, ""]
        
        # Environment section
        lines.extend([
            "ENVIRONMENT:",
            f"  env_id: {self.env_id}",
            f"    → Gymnasium environment identifier",
            f"  algo_id: {self.algo_id}",
            f"    → Reinforcement learning algorithm to use",
            f"  seed: {self.seed}",
            f"    → Random seed for reproducibility",
            ""
        ])
        
        # Networks section
        hidden_dims_str = str(self.hidden_dims) if isinstance(self.hidden_dims, tuple) else f"({self.hidden_dims},)"
        lines.extend([
            "NEURAL NETWORKS:",
            f"  hidden_dims: {hidden_dims_str}",
            f"    → Hidden layer dimensions for policy and value networks",
            f"  policy_lr: {self.policy_lr}",
            f"    → Learning rate for policy network optimizer",
            f"  value_lr: {self.value_lr}",
            f"    → Learning rate for value network optimizer",
            f"  ent_coef: {self.ent_coef}",
            f"    → Entropy regularization coefficient for exploration",
            ""
        ])
        
        # Training section
        lines.extend([
            "TRAINING:",
            f"  n_steps: {self.n_steps}",
            f"    → Number of environment steps per training rollout",
            f"  batch_size: {self.batch_size}",
            f"    → Batch size for training updates",
            f"  train_rollout_interval: {self.train_rollout_interval}",
            f"    → Number of epochs between training rollouts",
            f"  max_epochs: {self.max_epochs}",
            f"    → Maximum training epochs (-1 for unlimited)",
            f"  gamma: {self.gamma}",
            f"    → Discount factor for future rewards",
            f"  gae_lambda: {self.gae_lambda}",
            f"    → GAE lambda parameter for advantage estimation",
            f"  clip_range: {self.clip_range}",
            f"    → PPO clipping parameter for policy updates",
            ""
        ])
        
        # Evaluation section
        eval_episodes_str = "None" if self.eval_rollout_episodes is None else str(self.eval_rollout_episodes)
        eval_steps_str = "None" if self.eval_rollout_steps is None else str(self.eval_rollout_steps)
        lines.extend([
            "EVALUATION:",
            f"  eval_rollout_interval: {self.eval_rollout_interval}",
            f"    → Number of epochs between evaluation runs",
            f"  eval_rollout_episodes: {eval_episodes_str}",
            f"    → Number of episodes per evaluation (None = use steps)",
            f"  eval_rollout_steps: {eval_steps_str}",
            f"    → Number of steps per evaluation (None = use episodes)",
            f"  eval_async: {self.eval_async}",
            f"    → Whether to run evaluation asynchronously",
            ""
        ])
        
        # Normalization section
        lines.extend([
            "NORMALIZATION:",
            f"  normalize_obs: {self.normalize_obs}",
            f"    → Whether to normalize observations",
            f"  normalize_reward: {self.normalize_reward}",
            f"    → Whether to normalize rewards",
            ""
        ])
        
        # Miscellaneous section
        lines.extend([
            "MISCELLANEOUS:",
            f"  mean_reward_window: {self.mean_reward_window}",
            f"    → Window size for computing rolling mean reward",
        ])
        
        return "\n".join(lines)

def load_config(env_id: str, algo_id: str, config_dir: str = "configs") -> RLConfig:
    """Convenience function to load configuration."""
    return RLConfig.load_from_yaml(env_id, algo_id, config_dir)
