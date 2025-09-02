"""Configuration loading for environment YAML and legacy hyperparams."""

from dataclasses import MISSING, asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from .dict_utils import convert_dict_numeric_strings

def _dataclass_defaults_dict(cls: type) -> Dict[str, Any]:
    """Collect dataclass default values without instantiation."""
    defaults: Dict[str, Any] = {}
    for f in cls.__dataclass_fields__.values():
        if f.default is not MISSING: defaults[f.name] = f.default
        elif f.default_factory is not MISSING: defaults[f.name] = f.default_factory()  # type: ignore
    return defaults


@dataclass
class Config:
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
    obs_type: "Config.ObsType" = ObsType.rgb  # type: ignore[assignment]

    hidden_dims: Union[int, Tuple[int, ...]] = (64, 64)
    policy: "Config.PolicyType" = PolicyType.mlp  # type: ignore[assignment]
    policy_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"activation": "tanh"})

    policy_lr: float = 3e-4
    policy_lr_schedule: Optional[str] = None

    max_grad_norm: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: Optional[float] = 0.2
    clip_range_schedule: Optional[str] = None

    returns_type: "Config.ReturnsType" = ReturnsType.mc_episode
    normalize_returns: Optional[str] = None

    advantages_type: Optional["Config.AdvantagesType"] = None

    normalize_advantages: Optional["Config.AdvantageNormType"] = None

    reinforce_policy_targets: Optional["Config.ReinforceTargetsType"] = ReinforceTargetsType.returns  # type: ignore[assignment]

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

    accelerator: "Config.AcceleratorType" = AcceleratorType.cpu  # type: ignore[assignment]
    devices: Optional[Union[int, str]] = None

    quiet: bool = False

    @classmethod
    def load_from_yaml(cls, config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> 'Config':
        """Load config from environment YAMLs"""
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        env_config_path = project_root / config_dir

        # Build an index of config_id -> raw mapping supporting BOTH formats
        all_configs: Dict[str, Dict[str, Any]] = {}

        def _collect_from_file(path: Path) -> None:
            # Load the YAML file
            with open(path, "r", encoding="utf-8") as f: doc = yaml.safe_load(f) or {}

            # Load the base config from the YAML file
            config_field_names = set(cls.__dataclass_fields__.keys())
            base_config: Dict[str, Any] = {k: v for k, v in doc.items() if k in config_field_names}
            project_id = base_config.get("project_id", path.stem)

            # Search for variant configs in the YAML file
            # and add them to the all_configs dictionary 
            # (they inherit from base config)
            for k, v in doc.items():
                # Skip base config fields
                if k in config_field_names: continue

                # Skip non-dict fields
                if not isinstance(v, dict): continue

                # Create the variant config
                variant_id = str(k)
                variant_cfg = dict(base_config)
                variant_cfg.update(v)
                variant_cfg["project_id"] = project_id
                variant_config_id = f"{project_id}_{variant_id}"
                all_configs[variant_config_id] = variant_cfg

        # Load all config files
        yaml_files = sorted(env_config_path.glob("*.yaml"))
        yaml_files = [yf for yf in yaml_files if not yf.name.endswith(".spec.yaml")]
        for yf in yaml_files: _collect_from_file(yf)

        config_variant_id = f"{config_id}_{variant_id}"
        config_variant_cfg = all_configs[config_variant_id]

        # Create dict with dataclass defaults
        final_config: Dict[str, Any] = _dataclass_defaults_dict(cls)

        # Apply config variant over dataclass defaults
        final_config.update(config_variant_cfg)

        # Convert numeric strings to numbers
        convert_dict_numeric_strings(final_config)

        # Parse schedule specifiers like lin_0.001
        Config._parse_schedules(final_config)

        # Create config instance and validate
        instance = cls(**final_config)
        instance.validate()

        # Return the validated config
        return instance

    @classmethod
    def _parse_schedules(cls, config: Dict[str, Any]) -> None:
        # Iterate over the dictionary items (as list) to avoid 
        # modifying the dictionary size during iteration
        for key, value in list(config.items()):
            # Skip non-string values    
            if not isinstance(value, str): continue

            # Skip non-linear schedule values
            val_lower = value.lower()
            if not val_lower.startswith('lin_'): continue
            
            # Set the schedule and value
            config[f"{key}_schedule"] = 'linear'
            config[key] = float(val_lower.split('lin_')[1])

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
