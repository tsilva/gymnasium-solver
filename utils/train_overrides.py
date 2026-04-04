"""CLI override parsing and application helpers for training flows."""

from __future__ import annotations

from dataclasses import fields


def parse_config_overrides(override_list):
    """Parse KEY=VALUE strings into a dict with basic type inference."""
    if not override_list:
        return {}

    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected KEY=VALUE")

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() in ("true", "false"):
            overrides[key] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            overrides[key] = float(value) if "." in value else int(value)
        else:
            overrides[key] = value

    return overrides


def apply_env_kwargs_overrides(config, override_list):
    """Merge KEY=VALUE overrides into config.env_kwargs."""
    if not override_list:
        return config

    overrides = parse_config_overrides(override_list)
    if not hasattr(config, "env_kwargs") or config.env_kwargs is None:
        config.env_kwargs = {}

    for key, value in overrides.items():
        config.env_kwargs[key] = value
        print(f"env_kwargs override applied: {key} = {value}")

    return config


def apply_config_overrides(config, overrides):
    """Apply dict overrides to dataclass-backed config objects."""
    if not overrides:
        return config

    valid_fields = {field_info.name for field_info in fields(config)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"Invalid config field: {key}. Not a valid Config attribute.")
        setattr(config, key, value)
        print(f"Override applied: {key} = {value}")

    return config


def apply_cli_overrides(config, args):
    """Apply CLI-driven overrides in runtime precedence order."""
    if getattr(args, "overrides", None):
        overrides_dict = parse_config_overrides(args.overrides)
        config = apply_config_overrides(config, overrides_dict)

    if getattr(args, "env_kwargs", None):
        config = apply_env_kwargs_overrides(config, args.env_kwargs)

    cli_max_env_steps = int(args.max_env_steps) if getattr(args, "max_env_steps", None) else None
    if cli_max_env_steps is not None:
        current = getattr(config, "max_env_steps", None)
        if current != cli_max_env_steps:
            print(f"Overriding max_env_steps: {current} → {cli_max_env_steps}")
        config.max_env_steps = cli_max_env_steps

    return config
