"""Configuration management utilities."""

from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

def _convert_numeric_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string representations of numbers back to numeric types."""
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or 'E' in value:
                    config_dict[key] = float(value)
            except Exception:
                pass
    return config_dict


@dataclass
class Config:
    # ===== Core identifiers (mandatory) =====
    # Gymnasium environment ID (e.g., 'CartPole-v1', 'ALE/Pong-v5')
    env_id: str
    # Algorithm identifier used to select the agent implementation (e.g., 'ppo', 'reinforce', 'qlearning')
    algo_id: str

    # ===== Training data collection (mandatory) =====
    # Number of environment steps to collect per rollout before an update
    n_steps: int
    # Batch size used when sampling from the collected rollout data
    batch_size: int

    # ===== Training loop (optional) =====
    # Number of passes (epochs) over the rollout buffer per training epoch
    n_epochs: int = 1
    # Maximum number of trainer epochs (-1 or None means unlimited; training may stop earlier via n_timesteps)
    max_epochs: Optional[int] = None
    # Optional cap on total environment timesteps processed; when reached, training stops
    n_timesteps: Optional[float] = None

    # ===== Reproducibility & vectorization (optional) =====
    # Random seed for envs, torch, numpy, etc.
    seed: int = 42
    # Number of parallel vectorized environments
    n_envs: int = 1
    # Force a specific vectorization backend: True=SubprocVecEnv, False=DummyVecEnv, None=auto
    subproc: Optional[bool] = None

    # ===== Environment preprocessing (optional) =====
    # List of custom wrapper names registered in EnvWrapperRegistry to apply to each env instance
    env_wrappers: list = field(default_factory=list)
    # Environment-specific keyword args forwarded to gym.make()/OCAtari
    env_kwargs: dict = field(default_factory=dict)
    # Enable observation normalization: False=off, True=running norm, 'static'=fixed statistics
    normalize_obs: bool = False
    # Enable reward normalization (currently for completeness; may be unused in some algorithms)
    normalize_reward: bool = False
    # Convert observations to grayscale
    grayscale_obs: bool = False
    # Resize observations to a specific shape
    resize_obs: bool = False
    # Stack the last N frames along the channel dimension (useful for pixel/ram-based envs)
    frame_stack: int = 1
    # Observation type for ALE environments: 'rgb' (default), 'ram', 'grayscale', or 'objects'
    obs_type: str = "rgb"

    # ===== Model architecture (optional) =====
    # Hidden layer dimensions for policy/value networks (tuple or single int)
    hidden_dims: Union[int, Tuple[int, ...]] = (64, 64)
    # Policy selection and kwargs
    # policy can be 'mlp' or 'cnn'
    policy: str = 'mlp'
    # Optional policy kwargs (dict). When using environment YAML, this can be a mapping.
    policy_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"activation": "tanh"})

    # ===== Optimization (optional) =====
    # Base learning rate for optimizer (used unless 'learning_rate' override/schedule is provided)
    policy_lr: float = 3e-4
    # Optional learning rate value that can be scheduled (e.g., set via 'lin_3e-4')
    learning_rate: Optional[float] = None
    # Learning rate schedule strategy: None or 'linear' (when using 'learning_rate')
    learning_rate_schedule: Optional[str] = None
    # Gradient clipping max norm (applied per optimizer step)
    max_grad_norm: float = 0.5

    # ===== Algorithm hyperparameters (optional; sensible defaults) =====
    # Discount factor for future rewards
    gamma: float = 0.99
    # GAE lambda parameter (advantage estimation smoothing); ignored by some algos
    gae_lambda: float = 0.95
    # Entropy coefficient encouraging exploration
    ent_coef: float = 0.01
    # Value function loss coefficient (used by PPO/actor-critic methods)
    vf_coef: float = 0.5
    # PPO clip range base value (used by PPO); also supports scheduling
    clip_range: Optional[float] = 0.2
    # PPO clip range schedule strategy: None or 'linear'
    clip_range_schedule: Optional[str] = None
    # Normalize the returns; 'off' means no normalization, 'baseline' means normalize by baseline, 'batch' means normalize by batch mean and std
    normalize_returns: str = "batch"
    # Advantage normalization behavior: 'off', 'rollout', or 'batch'
    normalize_advantages: str = "batch"

    # ===== Evaluation (optional; disabled unless eval_freq_epochs is set) =====
    # Run evaluation every N training epochs; None disables evaluation entirely
    eval_freq_epochs: Optional[int] = None
    # Minimum number of epochs to wait before starting evaluations
    # Example: eval_warmup_epochs=5 -> first eval at end of epoch 5, then every eval_freq_epochs
    eval_warmup_epochs: int = 0
    # Number of episodes to run during each evaluation window
    eval_episodes: Optional[int] = None
    # Record evaluation videos every N evaluation epochs; defaults to eval_freq_epochs when not set
    eval_recording_freq_epochs: Optional[int] = None
    # Whether to decouple evaluation from training using async workers (not yet implemented)
    eval_async: bool = False
    # Use deterministic actions during evaluation (greedy for stochastic policies)
    eval_deterministic: bool = False
    # Optional target reward threshold to drive early-stopping or checkpointing heuristics
    reward_threshold: Optional[float] = None
    # Enable early stopping when eval mean reward reaches threshold
    early_stop_on_eval_threshold: bool = True
    # Enable early stopping when training mean episode reward reaches threshold
    early_stop_on_train_threshold: bool = False
    # Control verbosity of evaluation metric logging: when False, suppress per_env/* metrics in logs
    # (evaluate_policy still computes/returns them; this only affects logging)
    log_per_env_eval_metrics: bool = False

    # ===== Experiment tracking (optional) =====
    # Project identifier for logging (e.g., W&B project); defaults to a name derived from env_id
    project_id: Optional[str] = None
    # Directory for saving/loading checkpoints (used by custom checkpoint callback)
    checkpoint_dir: str = "checkpoints"
    # If True, attempt to resume from the latest checkpoint for this algo/env
    resume: bool = False

    # ===== Runtime / hardware (optional) =====
    # Device accelerator for PyTorch Lightning: 'cpu' or 'auto' (auto-detect GPU/MPS if available)
    accelerator: str = "cpu"
    # Number of devices to use or 'auto' (forwarded to Lightning as-is); None lets Lightning decide
    devices: Optional[Union[int, str]] = None

    # ===== CLI/UX (optional) =====
    # When True, run non-interactively (auto-accept prompts, suppress confirmations)
    quiet: bool = False

    # ===== Legacy compatibility (do not rely on these directly) =====
    # RL Zoo compatibility flag mapped to normalize_obs/reward
    normalize: Optional[bool] = None

    @classmethod
    def load_from_yaml(cls, config_id: str, algo_id: str = None, config_dir: str = "config/environments") -> 'Config':
        """
        Load configuration from YAML files supporting both formats:
        1. Environment-centric format: config_id is loaded from environment challenge files
        2. Legacy format: config_id is env_id, for backward compatibility with hyperparams folder
        
        If algo_id is None, it will be extracted from config_id (assuming format like "env_challenge_algo")
        """
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        
        # Try new environment-centric format first
        env_config_path = project_root / config_dir

        # Build an index of config_id -> raw mapping supporting BOTH formats
        all_configs: Dict[str, Dict[str, Any]] = {}
        # Track variant order per project (YAML file) to allow default selection
        project_variants: Dict[str, list] = {}

        def _is_new_style(doc: Dict[str, Any]) -> bool:
            # Heuristic: top-level contains env_id and other Config fields (not nested under a name)
            if not isinstance(doc, dict):
                return False
            fields = set(cls.__dataclass_fields__.keys())
            return "env_id" in doc and any(k in fields for k in doc.keys())

        def _collect_from_file(path: Path) -> None:
            # Ignore helper/example files with *.new.yaml extension
            if path.name.endswith(".new.yaml"):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                if not isinstance(doc, dict):
                    return
            except Exception:
                return

            if _is_new_style(doc):
                # New style: base fields at root + per-variant sections (e.g., ppo: {...})
                field_names = set(cls.__dataclass_fields__.keys())
                base: Dict[str, Any] = {k: v for k, v in doc.items() if k in field_names}
                project = base.get("project_id") or path.stem.replace(".new", "")
                # Maintain insertion-ordered list of variants for default selection
                if project not in project_variants:
                    project_variants[project] = []
                # Variant sections are top-level dict values whose key is not a dataclass field name
                for k, v in doc.items():
                    if k in field_names:
                        continue
                    if not isinstance(v, dict):
                        continue
                    # Record variant order for this project
                    project_variants[project].append(str(k))
                    variant_cfg = dict(base)
                    variant_cfg.update(v)
                    # Default algo_id to the variant section name when not provided
                    variant_cfg.setdefault("algo_id", str(k))
                    # Ensure a stable default project_id derived from the file name
                    # when it isn't explicitly provided in the YAML. This lets us
                    # omit redundant project_id entries that match the file's stem
                    # while preserving the same behavior.
                    variant_cfg.setdefault("project_id", project)
                    cid = f"{project}_{k}"
                    all_configs[cid] = variant_cfg
            else:
                # Old style: mapping of config_id -> mapping (may include inherits)
                for k, v in doc.items():
                    if isinstance(v, dict):
                        all_configs[str(k)] = v

        for yf in sorted(env_config_path.glob("*.yaml")):
            _collect_from_file(yf)

        if config_id in all_configs:
            return cls._load_from_environment_config(all_configs[config_id], all_configs)

        # Allow selecting by project/file name only for new-style configs
        # - If algo_id is provided, use <project>_<algo_id>
        # - Otherwise, pick the first variant declared in the YAML file
        if config_id in project_variants:
            if algo_id is not None:
                candidate = f"{config_id}_{algo_id}"
                if candidate in all_configs:
                    return cls._load_from_environment_config(all_configs[candidate], all_configs)
                raise ValueError(
                    f"Variant '{algo_id}' not found for project '{config_id}'. Available: {project_variants.get(config_id) or []}"
                )
            variants = project_variants.get(config_id) or []
            if variants:
                candidate = f"{config_id}_{variants[0]}"
                if candidate in all_configs:
                    return cls._load_from_environment_config(all_configs[candidate], all_configs)
        
        # Fall back to legacy format
        if algo_id is None:
            raise ValueError(f"Config '{config_id}' not found in environment configs and no algo_id provided for legacy format")
        
        config_path = project_root / "config/hyperparams"
        return cls._load_from_legacy_config(config_id, algo_id, config_path)

    @classmethod
    def _load_from_environment_config(cls, config_data: Dict[str, Any], all_configs: Dict[str, Any] = None) -> 'Config':
        """Load configuration from new environment-centric format with inheritance support."""
        # Start with class defaults
        final_config = {}
        for field in cls.__dataclass_fields__.values():
            if field.default is not MISSING:
                final_config[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore
                final_config[field.name] = field.default_factory()      # type: ignore

        # Handle inheritance
        if all_configs and 'inherits' in config_data:
            parent_config_id = config_data['inherits']
            if parent_config_id in all_configs:
                # Get parent config data (don't instantiate yet)
                parent_config_data = all_configs[parent_config_id]
                # Recursively resolve parent inheritance
                resolved_parent = cls._resolve_inheritance(parent_config_data, all_configs)
                # Apply parent settings
                final_config.update(resolved_parent)
            else:
                raise ValueError(f"Parent configuration '{parent_config_id}' not found for inheritance")

        # Apply current configuration settings (override parent)
        config_data_copy = config_data.copy()
        config_data_copy.pop('inherits', None)  # Remove inherits key from final config
        final_config.update(config_data_copy)

        # Convert any numeric strings (like scientific notation)
        final_config = _convert_numeric_strings(final_config)

        # Parse potential schedule specs before legacy compatibility so that override logic keeps initial
        cls._parse_schedules(final_config)

        # Handle RLZOO format compatibility
        cls._handle_legacy_compatibility(final_config)

        # Convert list values to tuples for hidden_dims
        if 'hidden_dims' in final_config and isinstance(final_config['hidden_dims'], list):
            final_config['hidden_dims'] = tuple(final_config['hidden_dims'])

        instance = cls(**final_config)
        instance._post_init_defaults()
        instance.validate()
        return instance

    @classmethod
    def _resolve_inheritance(cls, config_data: Dict[str, Any], all_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve inheritance chain without instantiating Config objects."""
        resolved_config = {}
        
        # Handle inheritance first
        if 'inherits' in config_data:
            parent_config_id = config_data['inherits']
            if parent_config_id in all_configs:
                parent_config_data = all_configs[parent_config_id]
                resolved_parent = cls._resolve_inheritance(parent_config_data, all_configs)
                resolved_config.update(resolved_parent)
            else:
                raise ValueError(f"Parent configuration '{parent_config_id}' not found for inheritance")
        
        # Apply current configuration settings (override parent)
        config_data_copy = config_data.copy()
        config_data_copy.pop('inherits', None)  # Remove inherits key
        resolved_config.update(config_data_copy)
        
        return resolved_config

    # (Dead code path removed)

    @classmethod
    def _load_from_legacy_config(cls, config_id: str, algo_id: str, config_path: Path) -> 'Config':
        """Load configuration from legacy hyperparams format."""
        # Start with class defaults
        final_config = {}
        for field in cls.__dataclass_fields__.values():
            if field.default is not MISSING:
                final_config[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore
                final_config[field.name] = field.default_factory()      # type: ignore

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
        cls._handle_legacy_compatibility(final_config)

        # Convert list values to tuples for hidden_dims
        if 'hidden_dims' in final_config and isinstance(final_config['hidden_dims'], list):
            final_config['hidden_dims'] = tuple(final_config['hidden_dims'])

        instance = cls(**final_config)
        instance._post_init_defaults()
        instance.validate()
        return instance

    @classmethod
    def _handle_legacy_compatibility(cls, config: Dict[str, Any]) -> None:
        """Handle RLZOO format compatibility."""
        # Use learning_rate if set, otherwise use policy_lr
        if 'learning_rate' in config and config['learning_rate'] is not None:
            config['policy_lr'] = config['learning_rate']
        
        # Handle normalize flag (RLZOO format) -> normalize_obs
        if 'normalize' in config and config['normalize'] is not None:
            config['normalize_obs'] = config['normalize']
            config['normalize_reward'] = config['normalize']
        
        if 'vf_coef' in config and config['vf_coef'] is not None:
            config['vf_coef'] = config['vf_coef']

    @classmethod
    def _parse_schedules(cls, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            value = config[key]
            if not isinstance(value, str): continue
            if not value.lower().startswith('lin_'): continue
            key_schedule = f"{key}_schedule"
            config[key_schedule] = 'linear'
            config[key] = float(value.lower().split('lin_')[1])

    # Derived defaults and cross-field normalization
    def _post_init_defaults(self) -> None:
        # If evaluation is enabled, ensure sensible episode/recording defaults
        if self.eval_freq_epochs is not None:
            if self.eval_episodes is None:
                self.eval_episodes = 10  # safe default evaluation horizon
            if self.eval_recording_freq_epochs is None:
                self.eval_recording_freq_epochs = self.eval_freq_epochs  # record when we evaluate
        # Map RL Zoo fields to internal ones if provided at construction-time
        if self.normalize is not None:
            self.normalize_obs = self.normalize
            self.normalize_reward = self.normalize
        # If a scheduled learning_rate is provided, prefer it for scheduling base
        # while keeping policy_lr as the optimizer's initial value
        if self.learning_rate is not None and self.policy_lr is None:
            self.policy_lr = self.learning_rate
        # Normalize policy name capitalization
        if isinstance(self.policy, str):
            self.policy = self.policy.strip()
        # Ensure policy_kwargs is a dict
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
        
    def rollout_collector_hyperparams(self) -> Dict[str, Any]:
        return {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'normalize_advantages': self.normalize_advantages == "rollout"
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
        if self.policy_lr is None or self.policy_lr <= 0:
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
        if self.n_timesteps is not None and self.n_timesteps <= 0:
            raise ValueError("n_timesteps must be a positive number when set.")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1].")
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.clip_range is not None and not (0 < self.clip_range < 1):
            raise ValueError("clip_range must be in (0, 1).")

        # Evaluation
        if self.eval_freq_epochs is not None and self.eval_freq_epochs <= 0:
            raise ValueError("eval_freq_epochs must be a positive integer when set.")
        if self.eval_warmup_epochs < 0:
            raise ValueError("eval_warmup_epochs must be a non-negative integer.")
        if self.eval_episodes is not None and self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be a positive integer when set.")
        if self.eval_recording_freq_epochs is not None and self.eval_recording_freq_epochs <= 0:
            raise ValueError("eval_recording_freq_epochs must be a positive integer when set.")
        if self.reward_threshold is not None and self.reward_threshold <= 0:
            raise ValueError("reward_threshold must be a positive float.")

        # Runtime / hardware
        allowed_accelerators = {"auto", "cpu", "gpu", "mps", "tpu", "ipu", "hpu"}
        if self.accelerator not in allowed_accelerators:
            raise ValueError(
                f"accelerator must be one of {sorted(allowed_accelerators)}; got '{self.accelerator}'"
            )
        if isinstance(self.devices, str) and self.devices != "auto":
            raise ValueError("devices may be an int, 'auto', or None")
        # Policy
        if isinstance(self.policy, str) and self.policy.lower() not in {"mlp", "cnn"}:
            raise ValueError("policy must be 'MLP' or 'CNN'")

    def save_to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        import json
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

def load_config(config_id: str, algo_id: str = None, config_dir: str = "config/environments") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(config_id, algo_id, config_dir)
