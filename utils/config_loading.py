"""Helpers for building Config instances from dicts and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from utils.io import read_yaml


def build_config_from_dict(
    config_dict: Dict[str, Any],
    *,
    algo_config_classes: Mapping[str, type],
):
    """Instantiate the correct config subclass from a raw config dict."""
    config_data = dict(config_dict)
    try:
        algo_id = config_data.pop("algo_id")
    except KeyError as exc:
        raise KeyError("config_dict must include 'algo_id'") from exc

    try:
        config_cls = algo_config_classes[str(algo_id)]
    except KeyError as exc:
        supported = ", ".join(sorted(algo_config_classes))
        raise ValueError(f"Unsupported algo_id '{algo_id}'. Expected one of: {supported}.") from exc

    valid_fields = {
        name
        for name, field_info in config_cls.__dataclass_fields__.items()
        if field_info.init
    }
    filtered_dict = {key: value for key, value in config_data.items() if key in valid_fields}
    return config_cls(**filtered_dict)


def load_config_from_yaml(
    *,
    config_id: str,
    variant_id: str | None,
    config_dir: str,
    project_root: Path,
    config_field_names: set[str],
    sanitize_name: Callable[[str], str],
    build_from_dict: Callable[[Dict[str, Any]], Any],
):
    """Load a config variant from the repository YAML files."""
    if variant_id is None:
        raise ValueError("variant_id is required. Use load_config(env_id, variant_id).")

    env_config_path = project_root / config_dir
    if not env_config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {env_config_path}")

    all_configs: Dict[str, Dict[str, Any]] = {}

    def _collect_from_file(path: Path) -> None:
        doc = read_yaml(path) or {}

        base_config: Dict[str, Any] = {}
        base_section = doc.get("_base") if isinstance(doc.get("_base"), dict) else {}
        if isinstance(base_section, dict):
            base_config.update({key: value for key, value in base_section.items() if key in config_field_names})
        base_config.update({key: value for key, value in doc.items() if key in config_field_names})

        for key, value in doc.items():
            if key in config_field_names or not isinstance(value, dict):
                continue
            if isinstance(key, str) and key.startswith("_"):
                continue

            public_variant_id = str(key)
            variant_cfg = dict(base_config)
            variant_cfg.update(value)

            if "project_id" not in variant_cfg or not variant_cfg["project_id"]:
                env_id = variant_cfg.get("env_id", "")
                obs_type = variant_cfg.get("obs_type", "rgb")
                obs_type_str = obs_type.value if hasattr(obs_type, "value") else str(obs_type)
                variant_cfg["project_id"] = f"{env_id}_{obs_type_str}" if env_id else path.stem

            project_id = str(variant_cfg["project_id"])
            alias_keys = {f"{project_id}_{public_variant_id}", f"{sanitize_name(project_id)}_{public_variant_id}"}

            env_id = variant_cfg.get("env_id")
            if env_id:
                alias_keys.add(f"{env_id}_{public_variant_id}")
                alias_keys.add(f"{sanitize_name(str(env_id))}_{public_variant_id}")

            for alias in alias_keys:
                all_configs.setdefault(alias, variant_cfg)

    for yaml_file in sorted(env_config_path.glob("*.yaml")):
        _collect_from_file(yaml_file)

    lookup_keys = [
        f"{config_id}_{variant_id}",
        f"{sanitize_name(config_id)}_{variant_id}",
    ]
    for lookup_key in lookup_keys:
        config_variant_cfg = all_configs.get(lookup_key)
        if config_variant_cfg is not None:
            return build_from_dict(config_variant_cfg)

    raise KeyError(f"Unknown config '{config_id}:{variant_id}'.")
