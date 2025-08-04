"""Configuration management utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, MISSING
from typing import Union, Tuple, Dict, Any, Optional
from utils.misc import _convert_numeric_strings

# TODO: review this file again
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
    # TODO: not supported yet
    use_sde: bool = False  # Use State Dependent Exploration
    sde_sample_freq: int = -1  # SDE sample frequency
    policy_kwargs: str = None  # Policy kwargs as string

    # Evaluation
    eval_freq_epochs: int = 10
    eval_episodes: int = 10
    eval_async: bool = False  # Default: true (async evaluation)
    eval_deterministic: bool = True  # Default: true (deterministic evaluation)
    reward_threshold: Optional[float] = None  # Default: None (use environment's reward threshold)


    # Normalization
    normalize_obs: bool = False  # Default: false
    normalize_reward: bool = False  # Default: false
    
    # Frame stacking
    frame_stack: int = 1  # Default: 1 (no frame stacking)
    
    # Atari-specific settings
    obs_type: str = "rgb"  # Default: "rgb" (other options: "ram", "grayscale")

    use_baseline: bool = False  # Use baseline subtraction for REINFORCE (default: false)
    
    # Checkpointing and resuming
    resume: bool = False  # Whether to resume training from the latest checkpoint
    checkpoint_dir: str = "checkpoints"  # Directory to save/load checkpoints
    
    env_wrappers: list = field(default_factory=list)

    @classmethod
    def load_from_yaml(cls, config_id: str, algo_id: str, config_dir: str = "config/hyperparams") -> 'Config':
        """
        Load configuration from YAML files supporting both formats:
        1. New format: config_id is a config identifier, env_id is read from the config
        2. Legacy format: config_id is env_id, for backward compatibility
        3. Inheritance: config can inherit from another config using 'inherit_from' field
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

        # Set algo_id
        final_config['algo_id'] = algo_id

        # Helper function to resolve inheritance
        def resolve_config_with_inheritance(config_name: str, visited: set = None) -> Dict[str, Any]:
            if visited is None:
                visited = set()
            
            if config_name in visited:
                raise ValueError(f"Circular inheritance detected: {' -> '.join(visited)} -> {config_name}")
            
            if config_name not in algo_config:
                raise ValueError(f"Config '{config_name}' not found for inheritance in {algo_config_path}")
            
            config_data = algo_config[config_name].copy()
            
            # Handle inheritance
            if 'inherit_from' in config_data:
                parent_config_name = config_data.pop('inherit_from')
                visited.add(config_name)
                parent_config = resolve_config_with_inheritance(parent_config_name, visited)
                visited.remove(config_name)
                
                # Merge parent config with child config (child overrides parent)
                merged_config = parent_config.copy()
                merged_config.update(config_data)
                return merged_config
            
            return config_data

        # Try new format first (config_id with env_id field)
        if config_id in algo_config:
            config_data = resolve_config_with_inheritance(config_id)
            final_config.update(config_data)
            
            # In new format, env_id should be in the config data
            if 'env_id' not in config_data:
                raise ValueError(f"Config '{config_id}' missing required 'env_id' field in {algo_config_path}")
                
        else:
            # Try legacy format (config_id is actually env_id)
            # Check if any config has this env_id
            matching_configs = []
            for conf_id, conf_data in algo_config.items():
                if isinstance(conf_data, dict) and conf_data.get('env_id') == config_id:
                    matching_configs.append(conf_id)
            
            if len(matching_configs) == 1:
                # Found exactly one config for this env_id in new format
                config_data = resolve_config_with_inheritance(matching_configs[0])
                final_config.update(config_data)
            elif len(matching_configs) > 1:
                # Multiple configs for same env_id - user needs to be specific
                raise ValueError(f"Multiple configs found for environment '{config_id}': {matching_configs}. "
                               f"Please specify one of these config IDs instead of the environment ID.")
            else:
                # Check if it's a legacy format (env_id as top-level key without env_id field)
                legacy_configs = {}
                for conf_id, conf_data in algo_config.items():
                    if isinstance(conf_data, dict) and 'env_id' not in conf_data:
                        legacy_configs[conf_id] = conf_data
                
                if config_id in legacy_configs:
                    # Legacy format: config_id is env_id, no env_id field in config
                    config_data = resolve_config_with_inheritance(config_id)
                    final_config.update(config_data)
                    final_config['env_id'] = config_id
                else:
                    available_configs = list(algo_config.keys())
                    raise ValueError(f"Config/Environment '{config_id}' not found in {algo_config_path}. "
                                   f"Available configs: {available_configs}")

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
        if self.reward_threshold is not None and self.reward_threshold <= 0:
            raise ValueError("reward_threshold must be a positive float.")

def load_config(config_id: str, algo_id: str, config_dir: str = "config/hyperparams") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(config_id, algo_id, config_dir)


def load_metrics_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load metrics configuration from YAML file.
    
    Args:
        config_dir: Directory containing the metrics.yaml file
        
    Returns:
        Dictionary containing metrics configuration
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    metrics_config_path = project_root / config_dir / "metrics.yaml"
    
    if not metrics_config_path.exists():
        raise FileNotFoundError(f"Metrics config file not found: {metrics_config_path}")
    
    with open(metrics_config_path, 'r') as f:
        return yaml.safe_load(f)


def get_metric_precision_dict(metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Convert metrics config to precision dictionary format expected by StdoutMetricsTable.
    
    This function takes metric names without namespaces and expands them to include
    all common namespaces (train/, eval/, rollout/, time/).
    
    Args:
        metrics_config: Metrics configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary mapping full metric names (with namespaces) to precision values
    """
    if metrics_config is None:
        metrics_config = load_metrics_config()
    
    # Get default precision from global config
    default_precision = metrics_config.get('_global', {}).get('default_precision', 4)
    
    # Common namespaces where metrics can appear
    namespaces = ['train', 'eval', 'rollout', 'time']
    
    precision_dict = {}
    
    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue
            
        # Get precision for this metric, default to global default
        precision = metric_config.get('precision', default_precision)
        
        # If force_integer is True, precision should be 0
        if metric_config.get('force_integer', False):
            precision = 0
        
        # Add the metric without namespace (for backward compatibility)
        precision_dict[metric_name] = precision
        
        # Add the metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            precision_dict[full_metric_name] = precision
    
    return precision_dict


def get_metric_delta_rules(metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, callable]:
    """Convert metrics config delta rules to callable format expected by StdoutMetricsTable.
    
    Args:
        metrics_config: Metrics configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary mapping metric names to validation functions
    """
    if metrics_config is None:
        metrics_config = load_metrics_config()
    
    namespaces = ['train', 'eval', 'rollout', 'time']
    delta_rules = {}
    
    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue
            
        # Check if this metric has a delta rule
        delta_rule = metric_config.get('delta_rule')
        if not delta_rule:
            continue
            
        if delta_rule == "non_decreasing":
            rule_fn = lambda prev, curr: curr >= prev
        else:
            # Add other rule types as needed
            continue
        
        # Add the rule for the metric without namespace
        delta_rules[metric_name] = rule_fn
        
        # Add the rule for the metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            delta_rules[full_metric_name] = rule_fn
    
    return delta_rules


def get_algorithm_metric_rules(algo_id: str, metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, dict]:
    """Get algorithm-specific metric validation rules.
    
    Args:
        algo_id: Algorithm identifier (e.g., 'ppo', 'reinforce')
        metrics_config: Metrics configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary mapping metric names to rule configurations
    """
    if metrics_config is None:
        metrics_config = load_metrics_config()
    
    rules = {}
    namespaces = ['train', 'eval', 'rollout', 'time']
    
    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue
            
        # Check if this metric has algorithm-specific rules
        algorithm_rules = metric_config.get('algorithm_rules', {})
        if not algorithm_rules:
            continue
            
        # Check if there's a rule for this specific algorithm
        rule_config = algorithm_rules.get(algo_id.lower())
        if not rule_config:
            continue
            
        threshold = rule_config.get('threshold')
        condition = rule_config.get('condition')
        message = rule_config.get('message', 'Metric validation failed')
        level = rule_config.get('level', 'warning')
        
        # Create the validation function based on condition
        if condition == "less_than":
            check_fn = lambda value, thresh=threshold: value < thresh
        elif condition == "greater_than":
            check_fn = lambda value, thresh=threshold: value > thresh
        elif condition == "between":
            min_val = rule_config.get('min', float('-inf'))
            max_val = rule_config.get('max', float('inf'))
            check_fn = lambda value, min_v=min_val, max_v=max_val: min_v <= value <= max_v
        else:
            continue
        
        rule_dict = {
            'check': check_fn,
            'message': message,
            'level': level
        }
        
        # Add rules for metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            rules[full_metric_name] = rule_dict
    
    return rules
