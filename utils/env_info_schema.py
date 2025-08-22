from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ValidationError = Tuple[str, str]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_or_str(value: Any) -> bool:
    return isinstance(value, (int, float)) or isinstance(value, str)


@dataclass
class ChoiceBlock:
    values: Optional[Sequence[Any]] = None
    default: Any = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChoiceBlock":
        return cls(values=data.get("values"), default=data.get("default"))

    def validate(self, name: str) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(self.values, Sequence) or len(self.values) == 0:
            errors.append((f"{name}.values", "must be a non-empty list"))
        if self.default is None:
            errors.append((f"{name}.default", "must be present"))
        elif isinstance(self.values, Sequence) and self.default not in self.values:
            errors.append((f"{name}.default", "must be one of 'values'"))
        return errors


@dataclass
class EnvInfo:
    source: Optional[str] = None
    description: Optional[str] = None
    action_space: Optional[Mapping[str, Any]] = None
    observation_space: Optional[Mapping[str, Any]] = None
    rewards: Optional[Mapping[str, Any]] = None
    defaults: Optional[Mapping[str, Any]] = None
    versions: Optional[Mapping[str, Any]] = None
    modes: Optional[ChoiceBlock] = None
    difficulties: Optional[ChoiceBlock] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnvInfo":
        known_keys = {
            "source",
            "description",
            "action_space",
            "observation_space",
            "rewards",
            "defaults",
            "versions",
            "modes",
            "difficulties",
        }
        modes = ChoiceBlock.from_mapping(data["modes"]) if isinstance(data.get("modes"), Mapping) else None
        difficulties = ChoiceBlock.from_mapping(data["difficulties"]) if isinstance(data.get("difficulties"), Mapping) else None
        extras = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            source=data.get("source"),
            description=data.get("description"),
            action_space=data.get("action_space"),
            observation_space=data.get("observation_space"),
            rewards=data.get("rewards"),
            defaults=data.get("defaults"),
            versions=data.get("versions"),
            modes=modes,
            difficulties=difficulties,
            extras=extras,
        )

    @classmethod
    def load_yaml(cls, path: Path) -> "EnvInfo":
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, Mapping):
            # Store raw in extras to preserve context; validation will report
            return cls(extras={"raw": data})
        return cls.from_mapping(data)

    # --- Validation helpers (ported from the previous functional validator) ---
    @staticmethod
    def _validate_labels(labels: Any, expected_count: int) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(labels, Mapping):
            errors.append(("action_space.labels", "must be a mapping from integer indices to string labels"))
            return errors
        keys = list(labels.keys())
        if not all(_is_int(k) for k in keys):
            errors.append(("action_space.labels", "keys must be integers 0..N-1"))
        if expected_count > 0 and set(keys) != set(range(expected_count)):
            errors.append(("action_space.labels", f"keys must exactly be 0..{expected_count-1}, got {sorted(keys)}"))
        for k, v in labels.items():
            if not isinstance(v, str):
                errors.append((f"action_space.labels[{k}]", "label must be a string"))
        return errors

    @classmethod
    def _validate_action_space(cls, spec: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(spec, Mapping):
            return [("action_space", "must be a mapping")]
        if "discrete" in spec:
            n = spec.get("discrete")
            if not _is_int(n) or n <= 0:
                errors.append(("action_space.discrete", "must be a positive integer"))
            labels = spec.get("labels")
            if labels is not None:
                errors.extend(cls._validate_labels(labels, int(n) if _is_int(n) else 0))
            full_space = spec.get("full_space")
            if full_space is not None:
                if not isinstance(full_space, Mapping):
                    errors.append(("action_space.full_space", "must be a mapping"))
                else:
                    count = full_space.get("count")
                    if not _is_int(count) or count <= 0:
                        errors.append(("action_space.full_space.count", "must be a positive integer"))
                    if "enable_with" in full_space and not isinstance(full_space.get("enable_with"), str):
                        errors.append(("action_space.full_space.enable_with", "must be a string if present"))
        return errors

    @staticmethod
    def _validate_variant(name: str, variant: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(variant, Mapping):
            return [(f"observation_space.variants.{name}", "must be a mapping")]
        if "n" in variant:
            if not _is_int(variant.get("n")) or int(variant["n"]) <= 0:
                errors.append((f"observation_space.variants.{name}.n", "must be a positive integer"))
        elif "shape" in variant:
            shape = variant.get("shape")
            if not isinstance(shape, Sequence) or len(shape) == 0:
                errors.append((f"observation_space.variants.{name}.shape", "must be a non-empty list"))
            else:
                for idx, dim in enumerate(shape):
                    if not (_is_int(dim) and int(dim) > 0) and not isinstance(dim, str):
                        errors.append((f"observation_space.variants.{name}.shape[{idx}]", "each dim must be positive int or string placeholder"))
        else:
            errors.append((f"observation_space.variants.{name}", "must include either 'n' or 'shape'"))
        if "dtype" in variant and not isinstance(variant.get("dtype"), str):
            errors.append((f"observation_space.variants.{name}.dtype", "must be a string if present"))
        if "range" in variant:
            rng = variant.get("range")
            if not isinstance(rng, Sequence) or len(rng) != 2:
                errors.append((f"observation_space.variants.{name}.range", "must be a 2-element list"))
            else:
                if not _is_number_or_str(rng[0]) or not _is_number_or_str(rng[1]):
                    errors.append((f"observation_space.variants.{name}.range", "values must be numbers or strings"))
        return errors

    @classmethod
    def _validate_observation_space(cls, spec: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(spec, Mapping):
            return [("observation_space", "must be a mapping")]
        default = spec.get("default")
        variants = spec.get("variants")
        if not isinstance(default, str):
            errors.append(("observation_space.default", "must be a string"))
        if not isinstance(variants, Mapping):
            errors.append(("observation_space.variants", "must be a mapping of variant name to spec"))
            return errors
        if isinstance(default, str) and default not in variants:
            errors.append(("observation_space.default", f"default '{default}' not found in variants"))
        for name, variant in variants.items():
            errors.extend(cls._validate_variant(str(name), variant))
        return errors

    @staticmethod
    def _validate_rewards(spec: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(spec, Mapping):
            return [("rewards", "must be a mapping")]
        if "range" in spec:
            rng = spec.get("range")
            if not isinstance(rng, Sequence) or len(rng) != 2:
                errors.append(("rewards.range", "must be a 2-element list"))
            else:
                if not _is_number_or_str(rng[0]) or not _is_number_or_str(rng[1]):
                    errors.append(("rewards.range", "values must be numbers or strings"))
        if "threshold_solved" in spec and not _is_number_or_str(spec.get("threshold_solved")):
            errors.append(("rewards.threshold_solved", "must be a number or string"))
        if "components" in spec:
            components = spec.get("components")
            if not isinstance(components, Sequence):
                errors.append(("rewards.components", "must be a list if present"))
            else:
                for idx, comp in enumerate(components):
                    if not isinstance(comp, Mapping):
                        errors.append((f"rewards.components[{idx}]", "must be a mapping"))
                    else:
                        if "name" not in comp or not isinstance(comp.get("name"), str):
                            errors.append((f"rewards.components[{idx}].name", "must be a string"))
        return errors

    @staticmethod
    def _validate_defaults(spec: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(spec, Mapping):
            return [("defaults", "must be a mapping")]
        if "frameskip" in spec:
            fs = spec.get("frameskip")
            if not _is_int(fs):
                if not (isinstance(fs, Sequence) and len(fs) == 2 and all(_is_int(x) for x in fs)):
                    errors.append(("defaults.frameskip", "must be int or [int,int]"))
        for key in ("map_size", "obs_type", "render_mode"):
            if key in spec and not isinstance(spec.get(key), (str, int)):
                errors.append((f"defaults.{key}", "must be a string or int"))
        return errors

    @staticmethod
    def _validate_versions(spec: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not isinstance(spec, Mapping):
            return [("versions", "must be a mapping of version to description")]
        for ver, desc in spec.items():
            if not isinstance(ver, str):
                errors.append(("versions.key", "version keys must be strings like 'v1'"))
            if not isinstance(desc, str) or not desc.strip():
                errors.append((f"versions.{ver}", "description must be a non-empty string"))
        return errors

    def validate(self) -> List[ValidationError]:
        errors: List[ValidationError] = []
        # Required keys
        required_fields = [
            ("source", self.source),
            ("description", self.description),
            ("action_space", self.action_space),
            ("observation_space", self.observation_space),
            ("rewards", self.rewards),
            ("versions", self.versions),
        ]
        for key, value in required_fields:
            if value is None:
                errors.append((key, "missing required key"))

        # Top-level types
        if self.source is not None and not isinstance(self.source, str):
            errors.append(("source", "must be a string URL"))
        if self.description is not None and not isinstance(self.description, str):
            errors.append(("description", "must be a string"))

        # Nested structures
        if self.action_space is not None:
            errors.extend(self._validate_action_space(self.action_space))
        if self.observation_space is not None:
            errors.extend(self._validate_observation_space(self.observation_space))
        if self.rewards is not None:
            errors.extend(self._validate_rewards(self.rewards))
        if self.defaults is not None:
            errors.extend(self._validate_defaults(self.defaults))
        if self.versions is not None:
            errors.extend(self._validate_versions(self.versions))
        if self.modes is not None:
            errors.extend(self.modes.validate("modes"))
        if self.difficulties is not None:
            errors.extend(self.difficulties.validate("difficulties"))
        return errors


def validate_env_info(env: Mapping[str, Any]) -> List[ValidationError]:
    """Dataclass-backed validator for env_info mappings."""
    model = EnvInfo.from_mapping(env)
    return model.validate()
