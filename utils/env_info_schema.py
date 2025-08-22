from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


ValidationError = Tuple[str, str]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_or_str(value: Any) -> bool:
    return isinstance(value, (int, float)) or isinstance(value, str)


def _validate_labels(labels: Any, expected_count: int) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if not isinstance(labels, Mapping):
        errors.append(("action_space.labels", "must be a mapping from integer indices to string labels"))
        return errors
    # keys should be 0..expected_count-1
    keys = list(labels.keys())
    if not all(_is_int(k) for k in keys):
        errors.append(("action_space.labels", "keys must be integers 0..N-1"))
    if expected_count > 0 and set(keys) != set(range(expected_count)):
        errors.append(("action_space.labels", f"keys must exactly be 0..{expected_count-1}, got {sorted(keys)}"))
    for k, v in labels.items():
        if not isinstance(v, str):
            errors.append((f"action_space.labels[{k}]", "label must be a string"))
    return errors


def _validate_action_space(spec: Any) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if not isinstance(spec, Mapping):
        return [("action_space", "must be a mapping")] 

    if "discrete" in spec:
        n = spec.get("discrete")
        if not _is_int(n) or n <= 0:
            errors.append(("action_space.discrete", "must be a positive integer"))
        labels = spec.get("labels")
        if labels is not None:
            errors.extend(_validate_labels(labels, int(n) if _is_int(n) else 0))
        # optional full_space for ALE
        full_space = spec.get("full_space")
        if full_space is not None:
            if not isinstance(full_space, Mapping):
                errors.append(("action_space.full_space", "must be a mapping"))
            else:
                count = full_space.get("count")
                if not _is_int(count) or count <= 0:
                    errors.append(("action_space.full_space.count", "must be a positive integer"))
                # enable_with can be any string key used to enable
                if "enable_with" in full_space and not isinstance(full_space.get("enable_with"), str):
                    errors.append(("action_space.full_space.enable_with", "must be a string if present"))
    else:
        # We only validate discrete for now; allow other types to pass through
        pass
    return errors


def _validate_variant(name: str, variant: Any) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if not isinstance(variant, Mapping):
        return [(f"observation_space.variants.{name}", "must be a mapping")] 
    # Two primary shapes: either 'n' for discrete observations, or 'shape' for arrays
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
    # dtype if present should be string
    if "dtype" in variant and not isinstance(variant.get("dtype"), str):
        errors.append((f"observation_space.variants.{name}.dtype", "must be a string if present"))
    # range if present should be [min, max]
    if "range" in variant:
        rng = variant.get("range")
        if not isinstance(rng, Sequence) or len(rng) != 2:
            errors.append((f"observation_space.variants.{name}.range", "must be a 2-element list"))
        else:
            if not _is_number_or_str(rng[0]) or not _is_number_or_str(rng[1]):
                errors.append((f"observation_space.variants.{name}.range", "values must be numbers or strings"))
    return errors


def _validate_observation_space(spec: Any) -> List[ValidationError]:
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
        errors.extend(_validate_variant(str(name), variant))
    # Optional components inside variants are free-form; no strict validation here to avoid deleting data.
    return errors


def _validate_rewards(spec: Any) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if not isinstance(spec, Mapping):
        return [("rewards", "must be a mapping")] 
    # Optional 'range'
    if "range" in spec:
        rng = spec.get("range")
        if not isinstance(rng, Sequence) or len(rng) != 2:
            errors.append(("rewards.range", "must be a 2-element list"))
        else:
            if not _is_number_or_str(rng[0]) or not _is_number_or_str(rng[1]):
                errors.append(("rewards.range", "values must be numbers or strings"))
    # Optional 'threshold_solved' numeric
    if "threshold_solved" in spec and not _is_number_or_str(spec.get("threshold_solved")):
        errors.append(("rewards.threshold_solved", "must be a number or string"))
    # If components exist, ensure they are list of mappings with a name
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


def _validate_defaults(spec: Any) -> List[ValidationError]:
    # Defaults are environment-specific; perform minimal sanity checks
    errors: List[ValidationError] = []
    if not isinstance(spec, Mapping):
        return [("defaults", "must be a mapping")] 
    # frameskip may be int or [min,max]
    if "frameskip" in spec:
        fs = spec.get("frameskip")
        if not _is_int(fs):
            if not (isinstance(fs, Sequence) and len(fs) == 2 and all(_is_int(x) for x in fs)):
                errors.append(("defaults.frameskip", "must be int or [int,int]"))
    # map_size, obs_type, render_mode can be strings
    for key in ("map_size", "obs_type", "render_mode"):
        if key in spec and not isinstance(spec.get(key), (str, int)):
            errors.append((f"defaults.{key}", "must be a string or int"))
    return errors


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


def _validate_choice_block(name: str, spec: Any) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if not isinstance(spec, Mapping):
        return [(name, "must be a mapping with 'values' and 'default'")] 
    values = spec.get("values")
    default = spec.get("default")
    if not isinstance(values, Sequence) or len(values) == 0:
        errors.append((f"{name}.values", "must be a non-empty list"))
    if default is None:
        errors.append((f"{name}.default", "must be present"))
    elif isinstance(values, Sequence) and default not in values:
        errors.append((f"{name}.default", "must be one of 'values'"))
    return errors


def validate_env_info(env: Mapping[str, Any]) -> List[ValidationError]:
    """
    Validate a loaded env_info YAML (as a dict). Returns a list of (path, message) errors.
    The validator is intentionally permissive to avoid deleting/restricting data.
    """
    errors: List[ValidationError] = []

    # required top-level keys
    required_keys = ["source", "description", "action_space", "observation_space", "rewards", "versions"]
    for key in required_keys:
        if key not in env:
            errors.append((key, "missing required key"))

    # Key-type checks
    if "source" in env and not isinstance(env.get("source"), str):
        errors.append(("source", "must be a string URL"))
    if "description" in env and not isinstance(env.get("description"), str):
        errors.append(("description", "must be a string"))

    if "action_space" in env:
        errors.extend(_validate_action_space(env.get("action_space")))
    if "observation_space" in env:
        errors.extend(_validate_observation_space(env.get("observation_space")))
    if "rewards" in env:
        errors.extend(_validate_rewards(env.get("rewards")))
    if "defaults" in env:
        errors.extend(_validate_defaults(env.get("defaults")))
    if "versions" in env:
        errors.extend(_validate_versions(env.get("versions")))

    # Optional blocks for ALE
    if "modes" in env:
        errors.extend(_validate_choice_block("modes", env.get("modes")))
    if "difficulties" in env:
        errors.extend(_validate_choice_block("difficulties", env.get("difficulties")))

    return errors
