"""Utilities for working with environment specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

Scalar = Union[int, float, str]

from enum import Enum

class RenderMode(Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"

def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return int(float(stripped))
    raise ValueError(f"Expected integer-compatible value, got {value!r}")


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)
    raise ValueError(f"Expected float-compatible value, got {value!r}")


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Expected boolean-compatible value, got {value!r}")


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _is_numpy_scalar(value: Any) -> bool:
    import numpy as np
    return isinstance(value, np.generic)

def _coerce_scalar(value: Any) -> Scalar:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if _is_numpy_scalar(value):
        return value.item()
    raise ValueError(f"Expected scalar value, got {value!r}")


def _coerce_scalar_sequence(values: Any) -> Tuple[Scalar, ...]:
    if values is None:
        return tuple()
    if not _is_sequence(values):
        raise ValueError(f"Expected a sequence of scalar values, got {values!r}")
    return tuple(_coerce_scalar(v) for v in values)


def _coerce_range(values: Any, *, label: str) -> Optional[Tuple[Scalar, Scalar]]:
    if values is None:
        return None
    if not _is_sequence(values) or len(values) != 2:
        raise ValueError(f"{label} must be a 2-element sequence, got {values!r}")
    return (_coerce_scalar(values[0]), _coerce_scalar(values[1]))


def _coerce_label_key(key: Any) -> Union[int, str]:
    if isinstance(key, bool):
        raise ValueError("Boolean labels are not supported")
    if isinstance(key, int):
        return key
    if isinstance(key, float):
        if key.is_integer():
            return int(key)
        raise ValueError(f"Action label keys must be integers, got {key!r}")
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        return stripped
    raise ValueError(f"Unsupported label key type: {type(key)!r}")


@dataclass(frozen=True)
class WrapperConfig:
    name: str
    entry_point: str
    kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_value(cls, value: Any) -> "WrapperConfig":
        if isinstance(value, Mapping):
            name = value.get("name")
            entry_point = value.get("entry_point")
            kwargs = value.get("kwargs")
        else:
            name = getattr(value, "name", None)
            entry_point = getattr(value, "entry_point", None)
            kwargs = getattr(value, "kwargs", None)
        assert isinstance(name, str) and isinstance(entry_point, str), f"Wrapper specification requires 'name' and 'entry_point' strings, got {value!r}"
        assert kwargs is None or isinstance(kwargs, Mapping), "Wrapper kwargs must be a mapping if provided"
        return cls(name=name, entry_point=entry_point, kwargs=dict(kwargs) if kwargs else None)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name, "entry_point": self.entry_point}
        if self.kwargs:
            data["kwargs"] = dict(self.kwargs)
        return data


@dataclass(frozen=True)
class FullActionSpaceSpec:
    count: Optional[int] = None
    enable_with: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FullActionSpaceSpec":
        enable_with = data.get("enable_with")
        return cls(
            count=_coerce_optional_int(data.get("count")),
            enable_with=str(enable_with).strip() if enable_with is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.count is not None:
            data["count"] = int(self.count)
        if self.enable_with is not None:
            data["enable_with"] = self.enable_with
        return data


@dataclass(frozen=True)
class ActionSpaceSpec:
    discrete: Optional[int] = None
    labels: Dict[Union[int, str], str] = field(default_factory=dict)
    note: Optional[str] = None
    full_space: Optional[FullActionSpaceSpec] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ActionSpaceSpec":
        labels: Dict[Union[int, str], str] = {}
        if isinstance(data.get("labels"), Mapping):
            for key, value in data["labels"].items():
                labels[_coerce_label_key(key)] = str(value)
        full_space = None
        if isinstance(data.get("full_space"), Mapping):
            full_space = FullActionSpaceSpec.from_mapping(data["full_space"])
        return cls(
            discrete=_coerce_optional_int(data.get("discrete")),
            labels=labels,
            note=_optional_str(data.get("note")),
            full_space=full_space,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.discrete is not None:
            data["discrete"] = int(self.discrete)
        if self.labels:
            data["labels"] = {k: v for k, v in self.labels.items()}
        if self.note is not None:
            data["note"] = self.note
        if self.full_space is not None:
            full = self.full_space.as_dict()
            if full:
                data["full_space"] = full
        return data


@dataclass(frozen=True)
class ObservationComponent:
    name: Optional[str] = None
    range: Optional[Tuple[Scalar, Scalar]] = None
    values: Tuple[Scalar, ...] = tuple()
    description: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObservationComponent":
        rng = _coerce_range(data.get("range"), label="Observation component range") if "range" in data else None
        values = _coerce_scalar_sequence(data.get("values")) if "values" in data else tuple()
        return cls(
            name=_optional_str(data.get("name")),
            range=rng,
            values=values,
            description=_optional_str(data.get("description")),
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.name is not None:
            data["name"] = self.name
        if self.range is not None:
            data["range"] = [self.range[0], self.range[1]]
        if self.values:
            data["values"] = list(self.values)
        if self.description is not None:
            data["description"] = self.description
        return data


@dataclass(frozen=True)
class ObservationVariant:
    dtype: Optional[str] = None
    shape: Tuple[Scalar, ...] = tuple()
    range: Optional[Tuple[Scalar, Scalar]] = None
    note: Optional[str] = None
    n: Optional[int] = None
    components: Tuple[ObservationComponent, ...] = tuple()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObservationVariant":
        shape: Tuple[Scalar, ...] = tuple()
        if (shape_values := data.get("shape")) is not None:
            assert _is_sequence(shape_values), f"Observation shape must be a sequence, got {shape_values!r}"
            shape = tuple(_coerce_scalar(v) for v in shape_values)
        rng = _coerce_range(data.get("range"), label="Observation range") if "range" in data else None
        n = _coerce_optional_int(data.get("n")) if "n" in data else None
        components: Tuple[ObservationComponent, ...] = tuple()
        if (raw_components := data.get("components")) is not None:
            assert _is_sequence(raw_components), "Observation components must be a sequence of mappings"
            components = tuple(
                ObservationComponent.from_mapping(comp)
                for comp in raw_components
                if isinstance(comp, Mapping)
            )
        return cls(
            dtype=_optional_str(data.get("dtype")),
            shape=shape,
            range=rng,
            note=_optional_str(data.get("note")),
            n=n,
            components=components,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.shape:
            data["shape"] = list(self.shape)
        if self.dtype is not None:
            data["dtype"] = self.dtype
        if self.range is not None:
            data["range"] = [self.range[0], self.range[1]]
        if self.note is not None:
            data["note"] = self.note
        if self.n is not None:
            data["n"] = int(self.n)
        if self.components:
            data["components"] = [component.as_dict() for component in self.components]
        return data


@dataclass(frozen=True)
class ObservationSpaceSpec:
    default: Optional[str] = None
    variants: Dict[str, ObservationVariant] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObservationSpaceSpec":
        default = data.get("default")
        assert default is None or isinstance(default, str), "Observation default must be a string if provided"
        variants: Dict[str, ObservationVariant] = {}
        if isinstance(data.get("variants"), Mapping):
            for name, variant in data["variants"].items():
                assert isinstance(variant, Mapping), f"Observation variant '{name}' must be a mapping"
                variants[str(name)] = ObservationVariant.from_mapping(variant)
        return cls(default=_optional_str(default), variants=variants)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.default is not None:
            data["default"] = self.default
        if self.variants:
            data["variants"] = {name: variant.as_dict() for name, variant in self.variants.items()}
        return data


@dataclass(frozen=True)
class RewardComponent:
    name: Optional[str] = None
    sign: Optional[str] = None
    value: Optional[Scalar] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RewardComponent":
        value = data.get("value")
        return cls(
            name=_optional_str(data.get("name")),
            sign=_optional_str(data.get("sign")),
            value=_coerce_scalar(value) if value is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.name is not None:
            data["name"] = self.name
        if self.sign is not None:
            data["sign"] = self.sign
        if self.value is not None:
            data["value"] = self.value
        return data


@dataclass(frozen=True)
class RewardsSpec:
    per_step: Optional[Scalar] = None
    per_point: Optional[Scalar] = None
    per_block: Optional[Scalar] = None
    life_lost: Optional[Scalar] = None
    on_goal: Optional[Scalar] = None
    otherwise: Optional[Scalar] = None
    successful_dropoff: Optional[Scalar] = None
    illegal_action: Optional[Scalar] = None
    distribution: Optional[str] = None
    description: Optional[str] = None
    shaping: Optional[str] = None
    threshold_solved: Optional[Scalar] = None
    range: Optional[Tuple[Scalar, Scalar]] = None
    components: Tuple[RewardComponent, ...] = tuple()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RewardsSpec":
        range_pair = _coerce_range(data.get("range"), label="Reward range") if "range" in data else None
        components: Tuple[RewardComponent, ...] = tuple()
        if (components_raw := data.get("components")) is not None:
            assert _is_sequence(components_raw), "Reward components must be a sequence"
            components = tuple(
                RewardComponent.from_mapping(comp)
                for comp in components_raw
                if isinstance(comp, Mapping)
            )
        def _scalar(name: str) -> Optional[Scalar]:
            v = data.get(name)
            return _coerce_scalar(v) if v is not None else None

        return cls(
            per_step=_scalar("per_step"),
            per_point=_scalar("per_point"),
            per_block=_scalar("per_block"),
            life_lost=_scalar("life_lost"),
            on_goal=_scalar("on_goal"),
            otherwise=_scalar("otherwise"),
            successful_dropoff=_scalar("successful_dropoff"),
            illegal_action=_scalar("illegal_action"),
            distribution=_optional_str(data.get("distribution")),
            description=_optional_str(data.get("description")),
            shaping=_optional_str(data.get("shaping")),
            threshold_solved=_scalar("threshold_solved"),
            range=range_pair,
            components=components,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key in (
            "per_step",
            "per_point",
            "per_block",
            "life_lost",
            "on_goal",
            "otherwise",
            "successful_dropoff",
            "illegal_action",
            "threshold_solved",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.description is not None:
            data["description"] = self.description
        if self.distribution is not None:
            data["distribution"] = self.distribution
        if self.shaping is not None:
            data["shaping"] = self.shaping
        if self.range is not None:
            data["range"] = [self.range[0], self.range[1]]
        if self.components:
            data["components"] = [component.as_dict() for component in self.components]
        return data


@dataclass(frozen=True)
class ReturnsSpec:
    episodic: Optional[str] = None
    range: Optional[Tuple[Scalar, Scalar]] = None
    threshold_solved: Optional[Scalar] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReturnsSpec":
        threshold = data.get("threshold_solved")
        return cls(
            episodic=_optional_str(data.get("episodic")),
            range=_coerce_range(data.get("range"), label="Return range") if "range" in data else None,
            threshold_solved=_coerce_scalar(threshold) if threshold is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.episodic is not None:
            data["episodic"] = self.episodic
        if self.range is not None:
            data["range"] = [self.range[0], self.range[1]]
        if self.threshold_solved is not None:
            data["threshold_solved"] = self.threshold_solved
        return data


@dataclass(frozen=True)
class ChoiceSpec:
    values: Tuple[Scalar, ...] = tuple()
    default: Optional[Scalar] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChoiceSpec":
        assert (values_raw := data.get("values")) is not None, "ChoiceSpec requires a 'values' sequence"
        default = data.get("default")
        return cls(
            values=_coerce_scalar_sequence(values_raw),
            default=_coerce_scalar(default) if default is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.values:
            data["values"] = list(self.values)
        if self.default is not None:
            data["default"] = self.default
        return data


@dataclass(frozen=True)
class EnvSpec:
    """Structured environment specification combining Gymnasium and YAML metadata."""

    id: Optional[str] = None
    source: Optional[str] = None
    description: Optional[str] = None
    goal: Optional[str] = None
    entry_point: Optional[str] = None
    reward_threshold: Optional[float] = None
    nondeterministic: Optional[bool] = None
    max_episode_steps: Optional[int] = None
    order_enforce: Optional[bool] = None
    autoreset: Optional[bool] = None
    disable_env_checker: Optional[bool] = None
    apply_api_compatibility: Optional[bool] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    additional_wrappers: Tuple[WrapperConfig, ...] = field(default_factory=tuple)
    vector_entry_point: Optional[str] = None
    namespace: Optional[str] = None
    name: Optional[str] = None
    version: Optional[int] = None
    render_fps: Optional[int] = None

    render_mode: Optional[RenderMode] = None
    action_space: Optional[ActionSpaceSpec] = None
    observation_space: Optional[ObservationSpaceSpec] = None
    rewards: Optional[RewardsSpec] = None
    returns: Optional[ReturnsSpec] = None
    modes: Optional[ChoiceSpec] = None
    difficulties: Optional[ChoiceSpec] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnvSpec":
        assert isinstance(data, Mapping), "EnvSpec.from_mapping expects a mapping"

        action_space = ActionSpaceSpec.from_mapping(data["action_space"]) if isinstance(data.get("action_space"), Mapping) else None
        observation_space = ObservationSpaceSpec.from_mapping(data["observation_space"]) if isinstance(data.get("observation_space"), Mapping) else None
        rewards = RewardsSpec.from_mapping(data["rewards"]) if isinstance(data.get("rewards"), Mapping) else None
        returns = ReturnsSpec.from_mapping(data["returns"]) if isinstance(data.get("returns"), Mapping) else None
        modes = ChoiceSpec.from_mapping(data["modes"]) if isinstance(data.get("modes"), Mapping) else None
        difficulties = ChoiceSpec.from_mapping(data["difficulties"]) if isinstance(data.get("difficulties"), Mapping) else None

        kwargs_value = data.get("kwargs")
        if kwargs_value is not None:
            assert isinstance(kwargs_value, Mapping), "Environment kwargs must be a mapping if provided"
        kwargs = dict(kwargs_value) if kwargs_value else {}

        wrappers: Tuple[WrapperConfig, ...] = tuple()
        if (wrappers_value := data.get("additional_wrappers")) is not None:
            assert _is_sequence(wrappers_value), "additional_wrappers must be a sequence"
            wrappers = tuple(WrapperConfig.from_value(item) for item in wrappers_value)

        return cls(
            id=_optional_str(data.get("id")),
            source=_optional_str(data.get("source")),
            description=_optional_str(data.get("description")),
            goal=_optional_str(data.get("goal")),
            entry_point=_optional_str(data.get("entry_point")),
            reward_threshold=_coerce_optional_float(data.get("reward_threshold")),
            nondeterministic=_coerce_optional_bool(data.get("nondeterministic")),
            max_episode_steps=_coerce_optional_int(data.get("max_episode_steps")),
            order_enforce=_coerce_optional_bool(data.get("order_enforce")),
            autoreset=_coerce_optional_bool(data.get("autoreset")),
            disable_env_checker=_coerce_optional_bool(data.get("disable_env_checker")),
            apply_api_compatibility=_coerce_optional_bool(data.get("apply_api_compatibility")),
            kwargs=kwargs,
            additional_wrappers=wrappers,
            vector_entry_point=_optional_str(data.get("vector_entry_point")),
            namespace=_optional_str(data.get("namespace")),
            name=_optional_str(data.get("name")),
            version=_coerce_optional_int(data.get("version")),
            render_fps=_coerce_optional_int(data.get("render_fps")),
            render_mode=_optional_str(data.get("render_mode")),
            action_space=action_space,
            observation_space=observation_space,
            rewards=rewards,
            returns=returns,
            modes=modes,
            difficulties=difficulties,
        )

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key in ("id", "source", "description", "goal", "entry_point", "reward_threshold",
                    "nondeterministic", "max_episode_steps", "order_enforce", "autoreset",
                    "disable_env_checker", "apply_api_compatibility", "vector_entry_point",
                    "namespace", "name", "version", "render_fps"):
            if (value := getattr(self, key)) is not None:
                data[key] = value
        if self.render_mode is not None:
            data["render_mode"] = self.render_mode
        if self.kwargs:
            data["kwargs"] = dict(self.kwargs)
        if self.additional_wrappers:
            data["additional_wrappers"] = [wrapper.as_dict() for wrapper in self.additional_wrappers]
        if self.action_space and (action_dict := self.action_space.as_dict()):
            data["action_space"] = action_dict
        if self.observation_space and (obs_dict := self.observation_space.as_dict()):
            data["observation_space"] = obs_dict
        if self.rewards and (rewards_dict := self.rewards.as_dict()):
            data["rewards"] = rewards_dict
        if self.returns and (returns_dict := self.returns.as_dict()):
            data["returns"] = returns_dict
        if self.modes:
            data["modes"] = self.modes.as_dict()
        if self.difficulties:
            data["difficulties"] = self.difficulties.as_dict()
        return data

    def get_max_episode_steps(self) -> Optional[int]:
        return self.max_episode_steps if self.max_episode_steps and self.max_episode_steps > 0 else None

    def get_render_fps(self) -> Optional[int]:
        return self.render_fps if self.render_fps and self.render_fps > 0 else None

    def get_render_mode(self) -> Optional[RenderMode]:
        return self.render_mode

    def get_action_labels(self) -> Dict[Union[int, str], str]:
        if self.action_space is None:
            return {}
        return dict(self.action_space.labels)

    def get_reward_range(self) -> Optional[Sequence[Scalar]]:
        if self.rewards is None or self.rewards.range is None:
            return None
        return [self.rewards.range[0], self.rewards.range[1]]

    def get_return_range(self) -> Optional[Sequence[Scalar]]:
        if self.returns is None or self.returns.range is None:
            return None
        return [self.returns.range[0], self.returns.range[1]]

    def get_return_threshold(self) -> Optional[Scalar]:
        if self.returns is not None and self.returns.threshold_solved is not None:
            return self.returns.threshold_solved
        return None

    def as_yaml_payload(self) -> Dict[str, Any]:
        """Return a YAML-friendly representation of the specification."""
        return self.as_dict()
