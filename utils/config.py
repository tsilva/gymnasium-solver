"""Configuration loading for environment YAML and legacy hyperparams."""

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from utils.config_defaults import (
    resolve_atari_defaults,
    resolve_batch_size,
    resolve_defaults,
    resolve_eval_warmup_epochs,
    resolve_n_envs,
    resolve_numeric_strings,
    resolve_policy,
    resolve_retro_defaults,
    resolve_vizdoom_defaults,
)
from utils.config_loading import build_config_from_dict, load_config_from_yaml
from utils.config_schedules import (
    resolve_schedule_defaults,
    resolve_schedule_dicts,
    validate_schedules,
)
from utils.config_serialization import config_to_dict
from utils.config_validation import validate_common_config, validate_ppo_config
from utils.formatting import sanitize_name
from utils.io import write_json
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

    # Seeds for train/val/test environments
    seed_train: int = 42
    seed_val: int = 1042
    seed_test: int = 2042

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

    # Number of times to repeat each action (frame skip)
    # frame_skip=1 means no frames are skipped (action applied once per step)
    # frame_skip=N means repeat the action N times
    # None means unset; will be filled with Atari defaults when vectorization_mode='atari'
    frame_skip: Optional[int] = None

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

    # Model architecture preset (e.g., "mlp_small", "cnn_nature")
    # Specifies policy type, hidden dims, activation, and policy_kwargs
    model_id: Optional[str] = None

    # Internal fields populated from model_id during __post_init__
    _hidden_dims: Optional[Union[int, Tuple[int, ...]]] = field(default=None, init=False, repr=False)
    _activation: Optional[str] = field(default=None, init=False, repr=False)
    _policy_kwargs: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

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
        """Validate all hyperparameter schedule configurations."""
        validate_schedules(self)

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

    # Whether to run evaluation asynchronously (non-blocking)
    # (when enabled, evaluation runs in background and doesn't block training)
    eval_async: bool = False

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

    # Run specification to initialize weights from (for transfer learning)
    # Format: 'run_id' or 'run_id/checkpoint'
    # Checkpoint can be '@best', '@last', or 'epoch=N'
    # If no checkpoint specified, uses @best if available, otherwise @last
    # Downloads from W&B if not found locally
    # Examples: "abc123", "abc123/@best", "abc123/epoch=13", "@last/@best"
    init_from_run: Optional[str] = None

    @property
    def max_vec_steps(self) -> Optional[int]:
        """Computed property: max_env_steps converted to vectorized steps."""
        if self.max_env_steps is None:
            return None
        return self.max_env_steps // self.n_envs

    @property
    def hidden_dims(self) -> Tuple[int, ...]:
        """Resolved hidden dimensions from model_id."""
        assert self._hidden_dims is not None, "hidden_dims not resolved. Ensure __post_init__ ran."
        return self._hidden_dims

    @property
    def activation(self) -> str:
        """Resolved activation function from model_id."""
        assert self._activation is not None, "activation not resolved. Ensure __post_init__ ran."
        return self._activation

    @property
    def policy_kwargs(self) -> Dict[str, Any]:
        """Resolved policy kwargs from model_id."""
        assert self._policy_kwargs is not None, "policy_kwargs not resolved. Ensure __post_init__ ran."
        return self._policy_kwargs

    @classmethod
    def build_from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return build_config_from_dict(
            config_dict,
            algo_config_classes={
                "reinforce": REINFORCEConfig,
                "ppo": PPOConfig,
            },
        )

    @classmethod
    def build_from_yaml(cls, config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> 'Config':
        """Load config from environment YAMLs."""
        return load_config_from_yaml(
            config_id=config_id,
            variant_id=variant_id,
            config_dir=config_dir,
            project_root=Path(__file__).parent.parent,
            config_field_names=set(cls.__dataclass_fields__.keys()),
            sanitize_name=sanitize_name,
            build_from_dict=cls.build_from_dict,
        )

    def __post_init__(self):
        self._resolve_defaults()
        self._resolve_n_envs()
        self._resolve_atari_defaults()
        self._resolve_vizdoom_defaults()
        self._resolve_retro_defaults()
        self._resolve_numeric_strings()
        self._resolve_batch_size()
        self._resolve_eval_warmup_epochs()
        self._resolve_schedules()
        self._resolve_schedule_defaults()
        self._resolve_policy()
        self.validate()
        
    def _resolve_policy(self) -> None:
        resolve_policy(self)

    def _resolve_defaults(self) -> None:
        resolve_defaults(self)

    def _resolve_n_envs(self) -> None:
        """Resolve n_envs "auto" to cpu_count()."""
        resolve_n_envs(self)

    def _resolve_atari_defaults(self) -> None:
        """Apply Atari defaults when vectorization_mode='atari' and params are not set."""
        resolve_atari_defaults(self)

    def _resolve_vizdoom_defaults(self) -> None:
        """Apply VizDoom defaults when params are not explicitly set."""
        resolve_vizdoom_defaults(self)

    def _resolve_retro_defaults(self) -> None:
        """Apply Retro (stable-retro) defaults when params are not explicitly set."""
        resolve_retro_defaults(self)

    def _resolve_numeric_strings(self) -> None:
        resolve_numeric_strings(self)

    def _resolve_batch_size(self) -> None:
        resolve_batch_size(self)

    def _resolve_eval_warmup_epochs(self) -> None:
        """Resolve fractional eval_warmup_epochs to absolute epochs."""
        resolve_eval_warmup_epochs(self)

    def _resolve_schedules(self) -> None:
        resolve_schedule_dicts(self, {"policy_lr", "ent_coef"})

    def _resolve_schedule_defaults(self) -> None:
        resolve_schedule_defaults(self)

    def get_env_args(self) -> Dict[str, Any]:
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
            frame_skip=self.frame_skip,
            obs_type=self.obs_type,
            render_mode=None,
            vectorization_mode=self.vectorization_mode,
            record_video=False,
            record_video_kwargs={},
            env_kwargs=self.env_kwargs
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

    def get_rollout_collector_kwargs(self) -> Dict[str, Any]:
        """Get all kwargs needed for RolloutCollector initialization."""
        return {
            "n_steps": self.n_steps,
            **self.rollout_collector_hyperparams(),
        }
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        data = config_to_dict(self)
        write_json(path, data, indent=2, ensure_ascii=False, default=str)
    
    # TODO: figure out a way to softcode this
    def validate(self):
        validate_common_config(self, config_enum_cls=Config, logger=logger)

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
    clip_range_vf: Union[float, Dict[str, Any]] = 0.2
    target_kl: Optional[float] = None # TODO: 0.015 in spinning up?
    ent_coef: float = 0.0
    vf_coef: Union[float, Dict[str, Any]] = 0.5
    max_grad_norm: float = 0.5
    returns_type: "Config.ReturnsType" = Config.ReturnsType.gae_rtg
    advantages_type: "Config.AdvantagesType" = Config.AdvantagesType.gae
    policy_targets: "Config.PolicyTargetsType" = Config.PolicyTargetsType.advantages  # type: ignore[assignment]
    normalize_advantages: "Config.AdvantageNormType" = Config.AdvantageNormType.batch

    @property
    def algo_id(self) -> str:
        return "ppo"

    def _resolve_schedules(self) -> None:
        super()._resolve_schedules()
        resolve_schedule_dicts(self, {"vf_coef", "clip_range", "clip_range_vf"})

    def validate(self):
        super().validate()
        validate_ppo_config(self, config_enum_cls=Config)

def load_config(config_id: str, variant_id: str = None, config_dir: str = "config/environments") -> Config:
    """Convenience function to load configuration."""
    return Config.build_from_yaml(config_id, variant_id, config_dir)
