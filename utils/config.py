"""Configuration loading for environment YAML and legacy hyperparams."""

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from utils.formatting import sanitize_name
from utils.io import read_yaml, write_json
from utils.validators import ensure_in_range, ensure_non_negative, ensure_positive

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # TODO: move all these enums to external file and reuse them across the codebase wherever these strings are used
    class PolicyType(str, Enum):
        mlp = "mlp"
        cnn = "cnn"
        mlp_actorcritic = "mlp_actorcritic"
        cnn_actorcritic = "cnn_actorcritic"

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
        vector = "vector"
        rgb = "rgb"
        ram = "ram"
        objects = "objects"

    class OptimizerType(str, Enum):
        adam = "adam"
        adamw = "adamw"
        sgd = "sgd"

    # The id of this configuration (optional; defaults inferred by loaders)
    project_id: str = "" # TODO: make these mandatory

    # The id of the environment to train on
    env_id: str = ""  

    # Description of this configuration variant
    description: str = ""

    # Descriptive spec metadata for the environment (merges into EnvInfoWrapper)
    spec: Dict[str, Any] = field(default_factory=dict)

    # The number of steps to collect per rollout environment
    # (algorithm-specific defaults live in algo config classes)
    n_steps: Optional[int] = None

    # Size of each batch of data to use for each gradient update
    # (algorithm-specific defaults live in algo config classes)
    # When set to a float in (0, 1], it is interpreted as a fraction of the rollout size
    batch_size: [Union[int, float]] = None

    # The number of epochs to train on the same rollout data
    # (algorithm-specific defaults live in algo config classes)
    n_epochs: Optional[int] = None

    # Max epochs to train for (optional)
    max_epochs: Optional[int] = None

    # Max environment steps (frames) to train for (optional)
    # This is the total number of environment interactions, NOT vectorized steps.
    # For example, with n_envs=8 and max_env_steps=1M, training will run for 125k vec_steps.
    # Schedule parameters and early stopping are tied to this value.
    max_env_steps: Optional[int] = None

    # Max steps each episode can have (truncate episode lengths)
    max_episode_steps: Optional[int] = None
    
    # Experiment seed (for reproducibility)
    seed: int = 42

    # How many parallel environments are used to collect rollouts
    # Can be an int or "auto" (which resolves to cpu_count())
    n_envs: Union[int, str] = "auto"

    # TODO: pass in env_kwargs instead
    # Overrides the environment reward threshold for early stopping
    reward_threshold: Optional[float] = None # TODO: rename to env_reward_threshold

    # List of environment wrappers to apply to the environment
    # (eg: reward shapers, frame stacking, etc)
    env_wrappers: list = field(default_factory=list)

    # Additional kwargs to pass to the environment factory
    env_kwargs: dict = field(default_factory=dict)

    # Vectorization mode for parallel environments
    # - "auto": Automatically select based on environment (uses ALE atari for Atari RGB, sync otherwise)
    # - "atari": Use Atari native vectorization (only valid for Atari RGB environments)
    # - "sync": Synchronous vectorization (SyncVectorEnv)
    # - "async": Asynchronous vectorization with subprocesses (AsyncVectorEnv)
    # TODO: use enum instead
    vectorization_mode: Optional[str] = "auto"

    # How many N last observations to stack (N=1 means no stacking, only current observation)
    frame_stack: int = None

    # Number of frames to skip between actions (ALE environments)
    # None means unset; will be filled with Atari defaults when vectorization_mode='atari'
    frameskip: Optional[int] = None

    # Whether to normalize observations using running mean and variance
    normalize_obs: bool = False

    # Whether to convert observations to grayscale (if representing images)
    # None means unset; will be filled with Atari defaults when vectorization_mode='atari'
    grayscale_obs: Optional[bool] = None

    # Whether to resize observations to a fixed size (if representing images)
    # None means unset; will be filled with Atari defaults when vectorization_mode='atari'
    # Can be bool (True defaults to (84, 84)) or tuple
    resize_obs: Optional[Union[bool, Tuple[int, int]]] = None

    # The type of observations (vector, RGB, RAM, or objects)
    obs_type: "Config.ObsType" = ObsType.vector  # type: ignore[assignment]

    # TODO: call this policy_type
    # Whether to use an MLP-based policy or actor-critic
    policy: "Config.PolicyType" = PolicyType.mlp  # type: ignore[assignment]

    # The dimensions of the hidden layers in the MLP
    hidden_dims: Union[int, Tuple[int, ...]] = None

    # The activation function to use in the MLP
    activation: str = "relu"

    # Additional kwargs to pass to the policy factory
    policy_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {})

    # The learning rate for the policy (algo defaults in subclasses)
    # Can be a float or a schedule dict: {start: float, end: float, from: float, to: float, schedule: str}
    policy_lr: Optional[Union[float, Dict[str, Any]]] = None

    # Optimizer to use for policy updates
    optimizer: "Config.OptimizerType" = OptimizerType.adam  # type: ignore[assignment]

    # The maximum gradient norm for the policy
    max_grad_norm: Optional[float] = None
    
    # The discount factor for the rewards (algo defaults in subclasses)
    # NOTE: effective horizon is 1 / (1 - gamma),
    # consider frameskips (eg: in Pong-v4, frameskip=4, so with gamma=0.99, effective horizon is 1 / (1 - 0.99) * 4 = 100 * 4 = 400)
    gamma: Optional[float] = None

    # The entropy coefficient for the policy (algo defaults in subclasses)
    # Can be a float or a schedule dict: {start: float, end: float, from: float, to: float, schedule: str}
    ent_coef: Optional[Union[float, Dict[str, Any]]] = None

    def _set_schedule_attrs(self, param: str, schedule_type: str, start_value: float,
                           end_value: float, from_pos: float, to_pos: float) -> None:
        """Set all schedule-related attributes for a parameter."""
        setattr(self, param, start_value)
        setattr(self, f"{param}_schedule", schedule_type)
        setattr(self, f"{param}_schedule_start_value", start_value)
        setattr(self, f"{param}_schedule_end_value", end_value)
        setattr(self, f"{param}_schedule_start", from_pos)
        setattr(self, f"{param}_schedule_end", to_pos)

    def _default_schedule_attr(self, attr: str, default: Any) -> None:
        """Set a schedule attribute to default if it's None."""
        if getattr(self, attr, None) is None:
            setattr(self, attr, default)
    
    def _validate_positive(self, attr: str, allow_none: bool = True) -> None:
        """Validate that an attribute is positive."""
        ensure_positive(getattr(self, attr, None), attr, allow_none=allow_none)

    def _validate_non_negative(self, attr: str, allow_none: bool = True) -> None:
        """Validate that an attribute is non-negative."""
        ensure_non_negative(getattr(self, attr, None), attr, allow_none=allow_none)

    def _validate_range(self, attr: str, min_val: float, max_val: float,
                       inclusive_min: bool = True, inclusive_max: bool = True) -> None:
        """Validate that an attribute is in a specific range."""
        ensure_in_range(
            getattr(self, attr, None),
            attr,
            min_val,
            max_val,
            inclusive_min=inclusive_min,
            inclusive_max=inclusive_max,
        )

    def _validate_schedules(self) -> None:
        """Validate all hyperparameter schedule configurations.

        Schedule start/end positions can be:
        - Fractional values in [0, 1]: interpreted as fraction of max_env_steps
        - Values > 1: interpreted as absolute env_steps
        """
        schedule_suffix = "_schedule"
        for key in list(vars(self).keys()):
            if not key.endswith(schedule_suffix):
                continue
            schedule = getattr(self, key)
            if not schedule:
                continue

            param = key[: -len(schedule_suffix)]
            start_value = getattr(self, f"{param}_schedule_start_value", None)
            end_value = getattr(self, f"{param}_schedule_end_value", None)
            assert start_value is not None and end_value is not None, \
                f"{param}_schedule requires start and end values to be defined."

            start_pos = getattr(self, f"{param}_schedule_start", None) or 0.0
            end_pos = getattr(self, f"{param}_schedule_end", None)
            if end_pos is None:
                assert self.max_env_steps is not None, \
                    f"{param}_schedule requires max_env_steps or an explicit schedule_end value."
                end_pos = 1.0

            assert start_pos >= 0 and end_pos >= 0, f"{param}_schedule start/end must be non-negative."
            assert end_pos >= start_pos, f"{param}_schedule end must be >= start."
            assert not (self.max_env_steps is None and (start_pos <= 1.0 or end_pos <= 1.0)), \
                f"{param}_schedule uses fractional start/end positions but config.max_env_steps is not set."

    # How to calculate rollout returns (algo defaults in subclasses)
    returns_type: Optional["Config.ReturnsType"] = None  # type: ignore[assignment]

    # Whether to normalize the returns
    # (none, baseline, or rollout)
    normalize_returns: Optional["Config.NormalizeReturnsType"] = None

    # How to calculate the policy targets (algo defaults in subclasses)
    policy_targets: Optional["Config.PolicyTargetsType"] = None  # type: ignore[assignment]

    # How many epochs to wait before starting to evaluate
    # (eval_freq_epochs doesn't apply until these many epochs have passed)
    # When set to a float in (0, 1), it is interpreted as a fraction of total training progress
    eval_warmup_epochs: Union[int, float] = 0

    # How many episodes to evaluate the policy for each evaluation
    # (stats will be averaged over all episodes; the more episodes, the more reliable the stats)
    eval_episodes: int = 100

    # How often to evaluate the policy (how many training epochs between evaluations)
    eval_freq_epochs: Optional[int] = None

    # Whether to run evaluation deterministically
    # (when set, the selected actions will always be the most likely instead of sampling from policy)
    eval_deterministic: bool = False

    # Whether to stop training when the training reward threshold is reached
    # When set to a float, that value overrides the env spec's reward threshold
    early_stop_on_train_threshold: Union[bool, float] = False

    # Whether to stop training when the evaluation reward threshold is reached
    # When set to a float, that value overrides the env spec's reward threshold
    early_stop_on_eval_threshold: Union[bool, float] = True

    # The accelerator to use for training (eg: simple environments are faster on CPU, image environments are faster on GPU)
    accelerator: "Config.AcceleratorType" = AcceleratorType.auto  # type: ignore[assignment]

    # The number of devices to use for training (eg: GPU, CPU)
    devices: Optional[Union[int, str]] = None

    # Whether to prompt the user before training starts
    quiet: bool = False

    # Whether to enable Weights & Biases logging
    enable_wandb: bool = True

    # Plateau intervention configuration (optional)
    # When a metric plateaus, cycle through parameter adjustments
    # Example: {"monitor": "train/roll/ep_rew/mean", "patience": 20, "actions": [...]}
    plateau_interventions: Optional[Dict[str, Any]] = None

    @property
    def max_vec_steps(self) -> Optional[int]:
        """Computed property: max_env_steps converted to vectorized steps."""
        if self.max_env_steps is None:
            return None
        return self.max_env_steps // self.n_envs

    @classmethod
    def build_from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        _config_dict = config_dict.copy()
        algo_id = _config_dict.pop("algo_id")
        config_cls = {
            "reinforce": REINFORCEConfig,
            "ppo": PPOConfig,
        }[algo_id]
        config = config_cls(**_config_dict)
        return config

    @classmethod
    def build_from_yaml(cls, config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> 'Config':
        """Load config from environment YAMLs"""
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        env_config_path = project_root / config_dir

        # Build an index of config_id -> raw mapping supporting BOTH formats
        all_configs: Dict[str, Dict[str, Any]] = {}

        def _collect_from_file(path: Path) -> None:
            # Load the YAML file
            doc = read_yaml(path) or {}

            # Load the base config from the YAML file
            config_field_names = set(cls.__dataclass_fields__.keys())
            base_config: Dict[str, Any] = {}
            base_section = doc.get("_base") if isinstance(doc.get("_base"), dict) else {}
            if isinstance(base_section, dict):
                base_config.update({k: v for k, v in base_section.items() if k in config_field_names})
            base_config.update({k: v for k, v in doc.items() if k in config_field_names})

            # Search for variant configs in the YAML file
            # and add them to the all_configs dictionary
            # (they inherit from base config)
            for k, v in doc.items():
                # Skip base config fields
                if k in config_field_names: continue

                # Skip non-dict fields
                if not isinstance(v, dict): continue

                # Skip meta/utility sections (e.g., anchors) prefixed with underscore
                if isinstance(k, str) and k.startswith("_"):
                    continue

                # Create the variant config
                variant_id = str(k)
                variant_cfg = dict(base_config)
                # Filter out fields not in Config dataclass, but keep algo_id (needed by build_from_dict)
                variant_cfg.update({k: v for k, v in v.items() if k in config_field_names or k == "algo_id"})

                # Construct default project_id from env_id + obs_type at variant level
                # (each variant may have different env_id/obs_type)
                if "project_id" not in variant_cfg or not variant_cfg["project_id"]:
                    env_id = variant_cfg.get("env_id", "")
                    obs_type = variant_cfg.get("obs_type", "rgb")
                    # Handle both string and enum cases
                    obs_type_str = obs_type.value if hasattr(obs_type, 'value') else str(obs_type)
                    # Always append obs_type to env_id for project_id
                    if env_id:
                        variant_cfg["project_id"] = f"{env_id}_{obs_type_str}"
                    else:
                        variant_cfg["project_id"] = path.stem

                project_id = variant_cfg["project_id"]
                variant_config_id = f"{project_id}_{variant_id}"
                all_configs[variant_config_id] = variant_cfg

                # Allow lookups by raw env id and sanitized aliases so callers
                # can pass either "ALE/Pong-v5" or "ALE-Pong-v5" without
                # needing to duplicate configs per naming style.
                alias_keys = {variant_config_id}
                env_id = variant_cfg.get("env_id")
                if env_id:
                    alias_keys.add(f"{env_id}_{variant_id}")
                    alias_keys.add(f"{sanitize_name(env_id)}_{variant_id}")
                alias_keys.add(f"{sanitize_name(project_id)}_{variant_id}")

                for alias in alias_keys:
                    all_configs.setdefault(alias, variant_cfg)

        # Load all config files
        yaml_files = sorted(env_config_path.glob("*.yaml"))
        for yf in yaml_files: _collect_from_file(yf)

        # Support passing a fully qualified id like "CartPole-v1_ppo" in config_id
        chosen_id = f"{config_id}_{variant_id}"
        config_variant_cfg = all_configs[chosen_id]

        # Create and return the config instance
        instance = cls.build_from_dict(config_variant_cfg)
        return instance

    def __post_init__(self):
        self._resolve_defaults()
        self._resolve_n_envs()
        self._resolve_atari_defaults()
        self._resolve_numeric_strings()
        self._resolve_batch_size()
        self._resolve_eval_warmup_epochs()
        self._resolve_schedules()
        self._resolve_schedule_defaults()
        self._resolve_policy()
        self.validate()
        
    # TODO: cleanup
    def _resolve_policy(self) -> None:
        is_mlp_policy = self.policy in [self.PolicyType.mlp, self.PolicyType.mlp_actorcritic]   
        if is_mlp_policy and self.hidden_dims is None:
            self.hidden_dims = [256, 256]

        is_cnn_policy = self.policy in [self.PolicyType.cnn, self.PolicyType.cnn_actorcritic]
        if is_cnn_policy and self.policy_kwargs is None:
            self.policy_kwargs = {
                "hidden_dims": [256, 256],
                "channels": [32, 64, 64],
                "kernel_sizes": [8, 4, 3],
                "strides": [4, 2, 1],
            }

    def _resolve_defaults(self) -> None:
        for f in self.__dataclass_fields__.values():
            value = getattr(self, f.name)
            if value is not None: continue
            if f.default is not MISSING: setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING: setattr(self, f.name, f.default_factory())

    def _resolve_n_envs(self) -> None:
        """Resolve n_envs "auto" to cpu_count()."""
        if self.n_envs == "auto":
            self.n_envs = os.cpu_count() or 1

    def _resolve_atari_defaults(self) -> None:
        """Apply Atari defaults when vectorization_mode='atari' and params are not set.

        ALE native vectorization applies these transformations under the hood:
        - frameskip: 4
        - grayscale: True
        - img_height: 84
        - img_width: 84
        - stack_num: 4

        This method syncs config with these defaults so the config reflects actual behavior.
        """
        if self.vectorization_mode not in ("atari", "auto"):
            return

        # Only apply defaults for ALE RGB environments
        from utils.environment import _is_alepy_env_id
        if not _is_alepy_env_id(self.env_id):
            return
        if self.obs_type != Config.ObsType.rgb:
            return

        # Apply Atari defaults if not explicitly set
        if self.grayscale_obs is None:
            self.grayscale_obs = True
        if self.resize_obs is None:
            self.resize_obs = (84, 84)
        if self.frame_stack == 1:  # Default value
            self.frame_stack = 4
        if self.frameskip is None:
            self.frameskip = 4

    def _resolve_numeric_strings(self) -> None:
        for key, value in list(asdict(self).items()):
            if not isinstance(value, str): continue
            try: setattr(self, key, float(value))
            except: pass

    def _resolve_batch_size(self) -> None:
        # Set default batch size based on policy type if not specified
        if self.batch_size is None:
            policy_str = self.policy.value if hasattr(self.policy, 'value') else str(self.policy)
            if 'cnn' in policy_str:
                self.batch_size = 256
            else:  # mlp policies
                self.batch_size = 64

        batch_size = self.batch_size
        if batch_size > 1: return
        rollout_size = self.n_envs * self.n_steps
        new_batch_size = max(1, int(rollout_size * batch_size))
        self.batch_size = new_batch_size

    def _resolve_eval_warmup_epochs(self) -> None:
        """Resolve fractional eval_warmup_epochs to absolute epochs."""
        warmup = self.eval_warmup_epochs
        # Only resolve if warmup is in (0, 1) - fractional range
        if warmup <= 0 or warmup >= 1:
            return

        # Fractional warmup requires max_env_steps to be set
        assert self.max_env_steps is not None, \
            "Fractional eval_warmup_epochs requires max_env_steps to be set"

        # Calculate total epochs: max_env_steps / (n_envs * n_steps)
        total_epochs = self.max_env_steps / (self.n_envs * self.n_steps)

        # Convert fraction to absolute epochs
        self.eval_warmup_epochs = int(total_epochs * warmup)

    def _resolve_schedules(self) -> None:
        # Schedulable parameters that support dict syntax (base params only)
        schedulable_params = {'policy_lr', 'ent_coef'}

        # Iterate over attribute names to avoid mutating during iteration
        for key in list(vars(self).keys()):
            value = getattr(self, key)

            # Handle dict-based schedule syntax
            if isinstance(value, dict) and key in schedulable_params:
                schedule_type = value.get('schedule', 'linear')
                start_value = value.get('start')
                end_value = value.get('end', 0.0)
                from_pos = value.get('from', 0.0)
                to_pos = value.get('to', 1.0)
                warmup = value.get('warmup', 0.0)

                assert start_value is not None, f"{key} schedule dict must have 'start' key"

                # Convert string values to floats (YAML may parse scientific notation as strings)
                start_value = float(start_value)
                end_value = float(end_value)
                from_pos = float(from_pos)
                to_pos = float(to_pos)
                warmup = float(warmup)

                self._set_schedule_attrs(key, schedule_type, start_value, end_value, from_pos, to_pos)
                # Set warmup if specified
                if warmup > 0.0:
                    setattr(self, f"{key}_schedule_warmup", warmup)

    def _resolve_schedule_defaults(self) -> None:
        schedule_suffix = "_schedule"
        for key in list(vars(self).keys()):
            if not key.endswith(schedule_suffix):
                continue
            schedule = getattr(self, key)
            if not schedule:
                continue

            param = key[: -len(schedule_suffix)]

            # Start/end values default to the current parameter value and zero respectively
            self._default_schedule_attr(f"{param}_schedule_start_value", getattr(self, param))
            self._default_schedule_attr(f"{param}_schedule_end_value", 0.0)

            # Default start/end positions use fractions of training progress
            self._default_schedule_attr(f"{param}_schedule_start", 0.0)
            self._default_schedule_attr(f"{param}_schedule_end", 1.0)

    def get_env_args(self) -> Dict[str, Any]:
        # Copy env_kwargs and add frameskip if set
        env_kwargs = dict(self.env_kwargs)
        if self.frameskip is not None:
            env_kwargs['frameskip'] = self.frameskip

        return dict(
            env_id=self.env_id,
            project_id=self.project_id,
            env_spec=self.spec,
            n_envs=self.n_envs,
            seed=self.seed,
            max_episode_steps=self.max_episode_steps,
            env_wrappers=self.env_wrappers,
            grayscale_obs=self.grayscale_obs,
            resize_obs=self.resize_obs,
            normalize_obs=self.normalize_obs,
            frame_stack=self.frame_stack,
            obs_type=self.obs_type,
            render_mode=None,
            vectorization_mode=self.vectorization_mode,
            record_video=False,
            record_video_kwargs={},
            env_kwargs=env_kwargs
        )

    def rollout_collector_hyperparams(self) -> Dict[str, Any]:
        result = {
            'gamma': self.gamma,
            'normalize_returns': self.normalize_returns == "rollout",
            'returns_type': (self.returns_type.value if hasattr(self.returns_type, 'value') else self.returns_type),
        }
        # Add algo-specific params if present
        if hasattr(self, 'gae_lambda'):
            result['gae_lambda'] = self.gae_lambda
        if hasattr(self, 'advantages_type'):
            result['advantages_type'] = (self.advantages_type.value if hasattr(self.advantages_type, 'value') else self.advantages_type)
        if hasattr(self, 'normalize_advantages'):
            result['normalize_advantages'] = self.normalize_advantages == "rollout"
        return result
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        data = asdict(self)
        data["algo_id"] = self.algo_id # TODO: do this in serializer method instead
        write_json(path, data, indent=2, ensure_ascii=False, default=str)
    
    # TODO: figure out a way to softcode this
    def validate(self):
        self._validate_positive("seed", allow_none=False)
        self._validate_positive("n_envs", allow_none=False)
        self._validate_positive("policy_lr")
        self._validate_non_negative("ent_coef")
        self._validate_positive("n_epochs")
        self._validate_positive("n_steps")
        self._validate_positive("batch_size")
        self._validate_positive("max_env_steps")
        self._validate_positive("frameskip")
        self._validate_range("gamma", 0, 1)
        self._validate_positive("eval_freq_epochs")
        self._validate_non_negative("eval_warmup_epochs", allow_none=False)
        self._validate_positive("eval_episodes")
        self._validate_positive("reward_threshold")

        # Validate and auto-round max_env_steps to be divisible by n_envs for clean conversion
        if self.max_env_steps is not None and self.max_env_steps % self.n_envs != 0:
            rounded = round(self.max_env_steps / self.n_envs) * self.n_envs
            logger.warning(
                f"max_env_steps ({self.max_env_steps}) not divisible by n_envs ({self.n_envs}). "
                f"Auto-rounding to {rounded}."
            )
            self.max_env_steps = rounded

        if self.devices is not None and not (isinstance(self.devices, int) or self.devices == "auto"):
            raise ValueError("devices may be an int, 'auto', or None.")

        # Validate vectorization_mode
        if self.vectorization_mode not in {"auto", "atari", "sync", "async", None}:
            raise ValueError(f"vectorization_mode must be 'auto', 'atari', 'sync', 'async', or None, got: {self.vectorization_mode}")

        # Validate that 'atari' is only used for Atari RGB environments
        if self.vectorization_mode == "atari":
            from utils.environment import _is_alepy_env_id
            if not _is_alepy_env_id(self.env_id):
                raise ValueError(f"vectorization_mode='atari' is only valid for Atari environments (ALE/*), got env_id: {self.env_id}")
            if self.obs_type != Config.ObsType.rgb:
                raise ValueError(f"vectorization_mode='atari' is only valid for RGB observations, got obs_type: {self.obs_type}")

        if self.n_envs is not None and self.n_steps is not None and self.batch_size is not None:
            rollout_size = self.n_envs * self.n_steps
            if not (self.batch_size <= rollout_size):
                raise ValueError(
                    f"batch_size ({self.batch_size}) should not exceed n_envs ({self.n_envs}) * n_steps ({self.n_steps})."
                )
            # Ensure uniform minibatches: batch_size must evenly divide rollout_size
            if rollout_size % int(self.batch_size) != 0:
                raise ValueError(
                    "batch_size must divide (n_envs * n_steps) exactly to yield uniform minibatches: "
                    f"rollout_size={rollout_size}, batch_size={self.batch_size}."
                )

        if self.policy_targets is not None and self.policy_targets not in {Config.PolicyTargetsType.returns, Config.PolicyTargetsType.advantages}:  # type: ignore[operator]
            raise ValueError("policy_targets must be 'returns' or 'advantages'.")

        # PPO replay-specific checks (only if fields exist)
        self._validate_non_negative("replay_ratio")
        self._validate_non_negative("replay_buffer_size")
        self._validate_positive("replay_is_clip")

        # Validate hyperparameter schedules
        self._validate_schedules()

# TODO: these config extensions should somehow be provided by the agent itself
@dataclass
class REINFORCEConfig(Config):
    policy: "Config.PolicyType" = Config.PolicyType.mlp  # type: ignore[assignment]
    n_steps: int = 2048
    batch_size: int = 2048
    n_epochs: int = 1
    policy_lr: float = 1e-2
    gamma: float = 0.99
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    returns_type: "Config.ReturnsType" = Config.ReturnsType.mc_rtg  # reward-to-go variant
    policy_targets: "Config.PolicyTargetsType" = Config.PolicyTargetsType.returns  # type: ignore[assignment]

    @property
    def algo_id(self) -> str:
        return "reinforce"

# TODO: default to 0.01 for atari if none specified
@dataclass
class PPOConfig(Config):
    policy: "Config.PolicyType" = Config.PolicyType.mlp_actorcritic  # type: ignore[assignment]
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    policy_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, Dict[str, Any]] = 0.2
    target_kl: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: Union[float, Dict[str, Any]] = 0.5
    max_grad_norm: float = 0.5
    returns_type: "Config.ReturnsType" = Config.ReturnsType.gae_rtg
    advantages_type: "Config.AdvantagesType" = Config.AdvantagesType.gae
    policy_targets: "Config.PolicyTargetsType" = Config.PolicyTargetsType.advantages  # type: ignore[assignment]
    normalize_advantages: "Config.AdvantageNormType" = Config.AdvantageNormType.rollout

    @property
    def algo_id(self) -> str:
        return "ppo"

    def _resolve_schedules(self) -> None:
        # PPO adds vf_coef and clip_range to schedulable params
        super()._resolve_schedules()

        schedulable_params = {'vf_coef', 'clip_range'}
        for key in list(vars(self).keys()):
            value = getattr(self, key)
            if isinstance(value, dict) and key in schedulable_params:
                schedule_type = value.get('schedule', 'linear')
                start_value = value.get('start')
                end_value = value.get('end', 0.0)
                from_pos = value.get('from', 0.0)
                to_pos = value.get('to', 1.0)
                warmup = value.get('warmup', 0.0)

                assert start_value is not None, f"{key} schedule dict must have 'start' key"

                start_value = float(start_value)
                end_value = float(end_value)
                from_pos = float(from_pos)
                to_pos = float(to_pos)
                warmup = float(warmup)

                self._set_schedule_attrs(key, schedule_type, start_value, end_value, from_pos, to_pos)
                if warmup > 0.0:
                    setattr(self, f"{key}_schedule_warmup", warmup)

    def validate(self):
        super().validate()

        # PPO-specific validations
        self._validate_positive("target_kl")
        self._validate_range("gae_lambda", 0, 1)
        self._validate_range("clip_range", 0, 1, inclusive_min=False, inclusive_max=False)
        self._validate_non_negative("vf_coef")

        if self.normalize_advantages is not None and self.normalize_advantages not in {Config.AdvantageNormType.rollout, Config.AdvantageNormType.batch, Config.AdvantageNormType.off}:  # type: ignore[operator]
            raise ValueError("normalize_advantages must be 'rollout', 'batch', or 'off'.")

def load_config(config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> Config:
    """Convenience function to load configuration."""
    return Config.build_from_yaml(config_id, variant_id, config_dir)
