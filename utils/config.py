"""Configuration loading for environment YAML and legacy hyperparams."""

from dataclasses import MISSING, asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

def _convert_numeric_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert scientific-notation strings back to numeric types (idempotent)."""
    for key, value in list(config_dict.items()):
        if isinstance(value, str):
            try:
                if 'e' in value.lower():
                    config_dict[key] = float(value)
            except Exception:
                # Leave value unchanged on parse failure
                pass
    return config_dict


def _dataclass_defaults_dict(cls: type) -> Dict[str, Any]:
    """Collect dataclass default values without instantiation."""
    defaults: Dict[str, Any] = {}
    for f in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore
            defaults[f.name] = f.default_factory()  # type: ignore
    return defaults


def _finalize_config_dict(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize a raw config dict prior to dataclass init (mutates and returns)."""
    # Numeric string conversions first
    _convert_numeric_strings(raw_config)
    # Parse schedule specifiers like lin_0.001
    Config._parse_schedules(raw_config)
    # RL Zoo compatibility mapping
    Config._handle_legacy_compatibility(raw_config)
    # Normalize hidden_dims to tuple when provided as list
    if isinstance(raw_config.get('hidden_dims'), list):
        raw_config['hidden_dims'] = tuple(raw_config['hidden_dims'])
    # Coerce and validate enum-like fields early
    Config._coerce_enums(raw_config)
    return raw_config


@dataclass
class Config:
    env_id: str
    algo_id: str

    n_steps: int
    batch_size: int

    n_epochs: int = 1
    project_id: Optional[str] = None
    max_epochs: Optional[int] = None
    n_timesteps: Optional[float] = None

    seed: int = 42
    n_envs: int = 1
    subproc: Optional[bool] = None

    env_wrappers: list = field(default_factory=list)
    env_kwargs: dict = field(default_factory=dict)
    normalize_obs: bool = False
    normalize_reward: bool = False
    grayscale_obs: bool = False
    resize_obs: bool = False
    frame_stack: int = 1
    obs_type: "Config.ObsType" = "rgb"  # type: ignore[assignment]

    hidden_dims: Union[int, Tuple[int, ...]] = (64, 64)
    policy: "Config.PolicyType" = "mlp"  # type: ignore[assignment]
    policy_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"activation": "tanh"})

    policy_lr: float = 3e-4
    learning_rate: Optional[float] = None
    learning_rate_schedule: Optional[str] = None
    max_grad_norm: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: Optional[float] = 0.2
    clip_range_schedule: Optional[str] = None

    returns_type: "Config.ReturnsType" = "montecarlo:episode"  # type: ignore[assignment]
    normalize_returns: Optional[str] = None

    advantages_type: Optional["Config.AdvantagesType"] = None

    normalize_advantages: Optional["Config.AdvantageNormType"] = None

    reinforce_policy_targets: Optional["Config.ReinforceTargetsType"] = "returns"  # type: ignore[assignment]

    eval_freq_epochs: Optional[int] = None
    eval_warmup_epochs: int = 0
    eval_episodes: Optional[int] = None
    eval_recording_freq_epochs: Optional[int] = None
    eval_async: bool = False
    eval_deterministic: bool = False
    reward_threshold: Optional[float] = None
    early_stop_on_eval_threshold: bool = True
    early_stop_on_train_threshold: bool = False
    log_per_env_eval_metrics: bool = False

    checkpoint_dir: str = "checkpoints"
    resume: bool = False

    accelerator: "Config.AcceleratorType" = "cpu"  # type: ignore[assignment]
    devices: Optional[Union[int, str]] = None

    quiet: bool = False

    normalize: Optional[bool] = None

    # ---- Enum helpers ----
    class PolicyType(str, Enum):
        mlp = "mlp"
        cnn = "cnn"

    class AcceleratorType(str, Enum):
        auto = "auto"
        cpu = "cpu"
        gpu = "gpu"
        mps = "mps"
        tpu = "tpu"
        ipu = "ipu"
        hpu = "hpu"

    class ReturnsType(str, Enum):
        mc_episode = "montecarlo:episode"
        mc_rtg = "montecarlo:reward_to_go"
        gae_rtg = "gae:reward_to_go"

    class AdvantagesType(str, Enum):
        gae = "gae"
        baseline_subtraction = "baseline_subtraction"

    class AdvantageNormType(str, Enum):
        rollout = "rollout"
        batch = "batch"
        off = "off"

    class ReinforceTargetsType(str, Enum):
        returns = "returns"
        advantages = "advantages"

    class ObsType(str, Enum):
        rgb = "rgb"
        ram = "ram"
        objects = "objects"

    @classmethod
    def load_from_yaml(cls, config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> 'Config':
        """Load config from environment YAMLs or legacy hyperparams."""
        # Normalize CLI-provided empty-string variant to None
        if isinstance(variant_id, str) and variant_id.strip() == "":
            variant_id = None

        # Get the project root directory
        project_root = Path(__file__).parent.parent
        
        # Try new environment-centric format first
        env_config_path = project_root / config_dir

        # Build an index of config_id -> raw mapping supporting BOTH formats
        all_configs: Dict[str, Dict[str, Any]] = {}
        # Track variant order per project (YAML file) to allow default selection
        project_variants: Dict[str, list] = {}

        def _is_new_style(doc: Dict[str, Any]) -> bool:
            return isinstance(doc, dict) and ("env_id" in doc)

        def _collect_from_file(path: Path) -> None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                if not isinstance(doc, dict):
                    return
            except Exception:
                return

            if _is_new_style(doc):
                field_names = set(cls.__dataclass_fields__.keys())
                base: Dict[str, Any] = {k: v for k, v in doc.items() if k in field_names}
                project = base.get("project_id") or path.stem.replace(".new", "")
                if project not in project_variants:
                    project_variants[project] = []
                for k, v in doc.items():
                    if k in field_names:
                        continue
                    if not isinstance(v, dict):
                        continue
                    project_variants[project].append(str(k))
                    variant_cfg = dict(base)
                    variant_cfg.update(v)
                    variant_cfg.setdefault("algo_id", str(k))
                    variant_cfg.setdefault("project_id", project)
                    cid = f"{project}_{k}"
                    all_configs[cid] = variant_cfg
            else:
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
            if variant_id is not None:
                candidate = f"{config_id}_{variant_id}"
                if candidate in all_configs:
                    return cls._load_from_environment_config(all_configs[candidate], all_configs)
                raise ValueError(
                    f"Variant '{variant_id}' not found for project '{config_id}'. Available: {project_variants.get(config_id) or []}"
                )
            variants = project_variants.get(config_id) or []
            if variants:
                candidate = f"{config_id}_{variants[0]}"
                if candidate in all_configs:
                    return cls._load_from_environment_config(all_configs[candidate], all_configs)
        
        # Fall back to legacy format
        if variant_id is None:
            raise ValueError(
                f"Config '{config_id}' not found in environment configs and no variant_id provided for legacy format"
            )
        
        config_path = project_root / "config/hyperparams"
        return cls._load_from_legacy_config(config_id, variant_id, config_path)

    @classmethod
    def _load_from_environment_config(cls, config_data: Dict[str, Any], all_configs: Dict[str, Any] = None) -> 'Config':
        """Load configuration from new environment-centric format with inheritance support."""
        # Start with class defaults
        final_config: Dict[str, Any] = _dataclass_defaults_dict(cls)

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

        _finalize_config_dict(final_config)
        return cls._instantiate(final_config)

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
        # Start with class defaults (once)
        final_config: Dict[str, Any] = _dataclass_defaults_dict(cls)

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

        _finalize_config_dict(final_config)
        return cls._instantiate(final_config)

    @classmethod
    def _instantiate(cls, final_config: Dict[str, Any]) -> 'Config':
        """Create Config instance and apply final normalization and validation."""
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
        # No-op fields are intentionally ignored

    @classmethod
    def _parse_schedules(cls, config: Dict[str, Any]) -> None:
        for key, value in list(config.items()):
            if not isinstance(value, str):
                continue
            val_lower = value.lower()
            if not val_lower.startswith('lin_'):
                continue
            config[f"{key}_schedule"] = 'linear'
            try:
                config[key] = float(val_lower.split('lin_')[1])
            except Exception:
                # Leave value unchanged on parse failure
                pass

    @classmethod
    def _coerce_enums(cls, config: Dict[str, Any]) -> None:
        """Normalize and validate fixed-choice fields using Enum definitions.
        Leaves values as canonical strings to avoid breaking existing code.
        """
        # policy -> Enum
        policy = config.get('policy')
        if isinstance(policy, str):
            pl = policy.strip().lower()
            if pl in (cls.PolicyType.mlp.value, cls.PolicyType.cnn.value):
                config['policy'] = cls.PolicyType(pl)
            else:
                raise ValueError("policy must be 'mlp' or 'cnn'")

        # accelerator -> Enum
        acc = config.get('accelerator')
        if isinstance(acc, str):
            al = acc.strip().lower()
            if al in set(a.value for a in cls.AcceleratorType):
                config['accelerator'] = cls.AcceleratorType(al)
            else:
                raise ValueError(f"accelerator must be one of {sorted([a.value for a in cls.AcceleratorType])}")

        # obs_type (for ALE/others) -> Enum
        obs = config.get('obs_type')
        if isinstance(obs, str):
            ol = obs.strip().lower()
            if ol in set(o.value for o in cls.ObsType):
                config['obs_type'] = cls.ObsType(ol)
            else:
                raise ValueError(f"obs_type must be one of {sorted([o.value for o in cls.ObsType])}")

        # normalize_advantages: allow bool or str aliases -> Enum
        na = config.get('normalize_advantages')
        if isinstance(na, bool):
            config['normalize_advantages'] = cls.AdvantageNormType.rollout if na else cls.AdvantageNormType.off
        elif isinstance(na, str):
            nal = na.strip().lower()
            if nal in set(n.value for n in cls.AdvantageNormType):
                config['normalize_advantages'] = cls.AdvantageNormType(nal)
            else:
                raise ValueError(f"normalize_advantages must be one of {sorted([n.value for n in cls.AdvantageNormType])} or a boolean")

        # normalize_returns: support 'rollout' (truthy) or 'off'/None -> keep as string flag for now
        nr = config.get('normalize_returns')
        if isinstance(nr, str):
            nrl = nr.strip().lower()
            if nrl in {cls.AdvantageNormType.rollout.value, cls.AdvantageNormType.off.value}:
                config['normalize_returns'] = nrl
            else:
                raise ValueError("normalize_returns must be 'rollout', 'off', or omitted")

        # returns_type with aliases; also set advantages_type when GAE requested -> Enum
        rt = config.get('returns_type')
        if isinstance(rt, str):
            rtl = rt.strip().lower()
            alias_map = {
                'episode': cls.ReturnsType.mc_episode,
                'reward_to_go': cls.ReturnsType.mc_rtg,
                'montecarlo:episode': cls.ReturnsType.mc_episode,
                'montecarlo:reward_to_go': cls.ReturnsType.mc_rtg,
                'gae': cls.ReturnsType.gae_rtg,
                'gae:reward_to_go': cls.ReturnsType.gae_rtg,
            }
            if rtl in alias_map:
                canonical_enum = alias_map[rtl]
                config['returns_type'] = canonical_enum
                if canonical_enum == cls.ReturnsType.gae_rtg:
                    # Ensure complementary advantages_type is set to GAE
                    adt = config.get('advantages_type')
                    if adt is None:
                        config['advantages_type'] = cls.AdvantagesType.gae
                    elif isinstance(adt, str) and adt.strip().lower() != cls.AdvantagesType.gae.value:
                        raise ValueError("advantages_type must be 'gae' when returns_type is 'gae:reward_to_go'")
            else:
                allowed = sorted(set(v.value for v in alias_map.values()))
                raise ValueError(f"returns_type must be one of {allowed} (aliases: 'episode', 'reward_to_go', 'gae')")

        # advantages_type when provided -> Enum
        adt = config.get('advantages_type')
        if isinstance(adt, str):
            adtl = adt.strip().lower()
            if adtl in set(a.value for a in cls.AdvantagesType):
                config['advantages_type'] = cls.AdvantagesType(adtl)
            else:
                raise ValueError(f"advantages_type must be one of {sorted([a.value for a in cls.AdvantagesType])}")

        # reinforce_policy_targets -> Enum
        rpt = config.get('reinforce_policy_targets')
        if isinstance(rpt, str):
            rpl = rpt.strip().lower()
            if rpl in set(r.value for r in cls.ReinforceTargetsType):
                config['reinforce_policy_targets'] = cls.ReinforceTargetsType(rpl)
            else:
                raise ValueError("reinforce_policy_targets must be 'returns' or 'advantages'")


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
        if isinstance(self.devices, str) and self.devices != "auto":
            raise ValueError("devices may be an int, 'auto', or None")

    def save_to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

def load_config(config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(config_id, variant_id, config_dir)
