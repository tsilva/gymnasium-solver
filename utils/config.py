"""Configuration loading for environment YAML and legacy hyperparams."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from .dict_utils import convert_dict_numeric_strings, dataclass_defaults_dict


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
        mc_episode = "mc:episode"
        mc_rtg = "mc:rtg"
        gae_rtg = "gae:rtg"

    class AdvantagesType(str, Enum):
        gae = "gae"
        baseline = "baseline"

    class AdvantageNormType(str, Enum):
        rollout = "rollout"
        batch = "batch"
        off = "off"

    class PolicyTargetsType(str, Enum):
        returns = "returns"
        advantages = "advantages"

    class ObsType(str, Enum):
        rgb = "rgb"
        ram = "ram"
        objects = "objects"

    # The id of this configuration
    project_id: str

    # The id of the environment to train on
    env_id: str

    # The id of the algorithm to train with
    algo_id: str

    # The number of steps to collect per rollout environment
    # (algorithm-specific defaults live in algo config classes)
    n_steps: Optional[int] = None

    # Size of each batch of data to use for each gradient update
    # (algorithm-specific defaults live in algo config classes)
    # When set to a float in (0, 1], it is interpreted as a fraction of the rollout size
    batch_size: [Union[int, float]] = 0.25

    # The number of epochs to train on the same rollout data
    # (algorithm-specific defaults live in algo config classes)
    n_epochs: Optional[int] = None

    # Max epochs to train for (optional)
    max_epochs: Optional[int] = None

    # Max timesteps to train for (optional)
    max_timesteps: Optional[int] = None

    # Max steps each episode can have (truncate episode lengths)
    max_episode_steps: Optional[int] = None
    
    # Experiment seed (for reproducibility)
    seed: int = 42

    # How many parallel environments are used to collect rollouts
    n_envs: int = 1

    # TODO: pass in env_kwargs instead
    # Overrides the environment reward threshold for early stopping
    reward_threshold: Optional[float] = None # TODO: rename to env_reward_threshold

    # List of environment wrappers to apply to the environment
    # (eg: reward shapers, frame stacking, etc)
    env_wrappers: list = field(default_factory=list)

    # Additional kwargs to pass to the environment factory
    env_kwargs: dict = field(default_factory=dict)

    # Whether to use subprocesses to run the parallel environments
    # (may slowdown or speedup depending on the environment)
    subproc: Optional[bool] = None

    # How many N last observations to stack (N=1 means no stacking, only current observation)
    frame_stack: int = 1

    # Whether to normalize observations using running mean and variance
    normalize_obs: bool = False

    # Whether to normalize rewards using running mean and variance
    normalize_reward: bool = False

    # Whether to convert observations to grayscale (if representing images)
    grayscale_obs: bool = False

    # Whether to resize observations to a fixed size (if representing images)
    resize_obs: bool = False

    # Whether the observations are RGB, RAM, or objects
    obs_type: "Config.ObsType" = ObsType.rgb  # type: ignore[assignment]

    # Whether to use a MLP or CNN policy
    policy: "Config.PolicyType" = PolicyType.mlp  # type: ignore[assignment]

    # The dimensions of the hidden layers in the MLP
    hidden_dims: Union[int, Tuple[int, ...]] = (64, 64)

    # Additional kwargs to pass to the policy factory
    policy_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"activation": "relu"})

    # The learning rate for the policy (algo defaults in subclasses)
    policy_lr: Optional[float] = None

    # The schedule for the policy learning rate
    policy_lr_schedule: Optional[str] = None

    # The maximum gradient norm for the policy
    max_grad_norm: Optional[float] = None
    
    # The discount factor for the rewards (algo defaults in subclasses)
    gamma: Optional[float] = None

    # The lambda parameter for the GAE (algo defaults in subclasses)
    gae_lambda: Optional[float] = None

    # The entropy coefficient for the policy (algo defaults in subclasses)
    ent_coef: Optional[float] = None

    # The value function coefficient for the policy (algo defaults in subclasses)
    vf_coef: Optional[float] = None

    # The clip range for the policy (algo defaults in subclasses)
    clip_range: Optional[float] = None

    # The schedule for the clip range
    clip_range_schedule: Optional[str] = None

    # How to calculate rollout returns (algo defaults in subclasses)
    returns_type: Optional["Config.ReturnsType"] = None  # type: ignore[assignment]

    # Whether to normalize the returns
    # (none, baseline, or rollout)
    normalize_returns: Optional["Config.NormalizeReturnsType"] = None

    # How to calculate rollout advantages (eg: GAE, Baseline Subtraction)
    # (none, gae, or baseline)
    advantages_type: Optional["Config.AdvantagesType"] = None

    # Whether to normalize the advantages
    # (none, rollout, or batch)
    normalize_advantages: Optional["Config.AdvantageNormType"] = None

    # How to calculate the policy targets for the REINFORCE algorithm
    # (algo defaults in subclass)
    policy_targets: Optional["Config.PolicyTargetsType"] = None  # type: ignore[assignment]

    # How many epochs to wait before starting to evaluate 
    # (eval_freq_epochs doesn't apply until these many epochs have passed)
    eval_warmup_epochs: int = 0

    # How often to evaluate the policy (how many training epochs between evaluations)
    eval_freq_epochs: Optional[int] = None

    # How many episodes to evaluate the policy for each evaluation
    # (stats will be averaged over all episodes; the more episodes, the more reliable the stats)
    eval_episodes: Optional[int] = None

    # How often to record videos of the policy during evaluation
    # (how many training epochs between recordings)
    eval_recording_freq_epochs: Optional[int] = None # TODO: pivot off N_evals instead

    # Whether to run evaluation deterministically
    # (when set, the selected actions will always be the most likely instead of sampling from policy)
    eval_deterministic: bool = False

    # Whether to stop training when the training reward threshold is reached
    early_stop_on_train_threshold: bool = False

    # Whether to stop training when the evaluation reward threshold is reached
    early_stop_on_eval_threshold: bool = True

    # The accelerator to use for training (eg: simple environments are faster on CPU, image environments are faster on GPU)
    accelerator: "Config.AcceleratorType" = AcceleratorType.cpu  # type: ignore[assignment]

    # The number of devices to use for training (eg: GPU, CPU)
    devices: Optional[Union[int, str]] = None

    # Whether to prompt the user before training starts
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

        # Select algorithm-specific config class based on algo_id
        algo_id = str(config_variant_cfg.get("algo_id", "")).lower()
        ConfigClass = {
            "qlearning": QLearningConfig,
            "reinforce": REINFORCEConfig,
            "ppo": PPOConfig,
        }.get(str(algo_id).lower(), Config)

        # Create dict with dataclass defaults from the selected class
        final_config: Dict[str, Any] = dataclass_defaults_dict(ConfigClass)

        # Apply config variant over dataclass defaults
        final_config.update(config_variant_cfg)

        # Convert numeric strings to numbers
        convert_dict_numeric_strings(final_config)

        # Parse schedule specifiers like lin_0.001
        Config._parse_schedules(final_config)

        # Support fractional batch_size: interpret 0 < batch_size <= 1 as a
        # fraction of the rollout size (n_envs * n_steps). This allows configs
        # to specify, e.g., 0.25 to mean 25% of the collected rollout per
        # gradient update.
        batch_size = final_config["batch_size"]
        if batch_size < 1:
            n_envs = final_config["n_envs"]
            n_steps = final_config["n_steps"]
            rollout_size = n_envs * n_steps
            assert rollout_size % (1 / batch_size) == 0, f"fractional batch_size must be a fraction of the rollout size: {rollout_size} % {1/batch_size} != 0"
            new_batch_size = int(rollout_size * batch_size)
            assert rollout_size % new_batch_size == 0, f"fractional batch_size must be a fraction of the rollout size: {rollout_size} % {new_batch_size} != 0"
            final_config["batch_size"] = new_batch_size

        # Create config instance and validate
        instance = ConfigClass(**final_config)
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
            'normalize_returns': self.normalize_returns == "rollout",
            'returns_type': self.returns_type,
            'advantages_type': self.advantages_type,
            'normalize_advantages': self.normalize_advantages == "rollout",
        }
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2, default=str)

    def validate(self):
        assert self.seed > 0, "seed must be a positive integer."
        assert self.policy_lr is None or self.policy_lr > 0, "policy_lr must be a positive float when set."
        assert self.ent_coef is None or self.ent_coef >= 0, "ent_coef must be a non-negative float when set."
        assert self.n_epochs is None or self.n_epochs > 0, "n_epochs must be a positive integer when set."
        assert self.n_steps is None or self.n_steps > 0, "n_steps must be a positive integer when set."
        assert self.batch_size is None or self.batch_size > 0, "batch_size must be a positive integer when set."
        assert self.max_timesteps is None or self.max_timesteps > 0, "max_timesteps must be a positive number when set."
        assert self.gamma is None or (0 < self.gamma <= 1), "gamma must be in (0, 1] when set."
        assert self.gae_lambda is None or (0 <= self.gae_lambda <= 1), "gae_lambda must be in [0, 1] when set."
        assert self.clip_range is None or (0 < self.clip_range < 1), "clip_range must be in (0, 1) when set."
        assert self.eval_freq_epochs is None or self.eval_freq_epochs > 0, "eval_freq_epochs must be a positive integer when set."
        assert self.eval_warmup_epochs >= 0, "eval_warmup_epochs must be a non-negative integer."
        assert self.eval_episodes is None or self.eval_episodes > 0, "eval_episodes must be a positive integer when set."
        assert self.eval_recording_freq_epochs is None or self.eval_recording_freq_epochs > 0, "eval_recording_freq_epochs must be a positive integer when set."
        assert self.reward_threshold is None or self.reward_threshold > 0, "reward_threshold must be a positive float when set."
        assert self.early_stop_on_train_threshold or self.early_stop_on_eval_threshold, "At least one of early_stop_on_train_threshold or early_stop_on_eval_threshold must be True."
        assert self.devices is None or isinstance(self.devices, int) or self.devices == "auto", "devices may be an int, 'auto', or None."
        assert self.batch_size <= self.n_envs * self.n_steps, f"batch_size ({self.batch_size}) should not exceed n_envs ({self.n_envs}) * n_steps ({self.n_steps})."
        assert self.policy_targets in {Config.PolicyTargetsType.returns, Config.PolicyTargetsType.advantages}, "policy_targets must be 'returns' or 'advantages'."

@dataclass
class QLearningConfig(Config):
    # Basic defaults for tabular Q-learning style collection
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 1
    gamma: float = 0.99

@dataclass
class REINFORCEConfig(Config):
    # Literature-style defaults for REINFORCE
    n_steps: int = 2048
    batch_size: int = 2048
    n_epochs: int = 1
    policy_lr: float = 1e-2
    gamma: float = 0.99
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    returns_type: "Config.ReturnsType" = Config.ReturnsType.mc_rtg  # reward-to-go variant
    policy_targets: "Config.PolicyTargetsType" = Config.PolicyTargetsType.returns  # type: ignore[assignment]


@dataclass
class PPOConfig(Config):
    # Literature/SB3-style defaults for PPO
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    policy_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    returns_type: "Config.ReturnsType" = Config.ReturnsType.gae_rtg
    advantages_type: "Config.AdvantagesType" = Config.AdvantagesType.gae
    policy_targets: "Config.PolicyTargetsType" = Config.PolicyTargetsType.advantages  # type: ignore[assignment]


def load_config(config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> Config:
    """Convenience function to load configuration."""
    return Config.load_from_yaml(config_id, variant_id, config_dir)
