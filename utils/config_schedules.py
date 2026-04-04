"""Schedule parsing and validation helpers for Config objects."""

from __future__ import annotations


def resolve_schedule_dicts(config, schedulable_params: set[str]) -> None:
    for key in list(vars(config).keys()):
        value = getattr(config, key)
        if not isinstance(value, dict) or key not in schedulable_params:
            continue

        schedule_type = value.get("schedule", "linear")
        start_value = value.get("start")
        end_value = value.get("end", 0.0)
        from_pos = value.get("from", 0.0)
        to_pos = value.get("to", 1.0)
        warmup = value.get("warmup", 0.0)

        assert start_value is not None, f"{key} schedule dict must have 'start' key"

        start_value = float(start_value)
        end_value = float(end_value)
        from_pos = float(from_pos)
        to_pos = float(to_pos)
        warmup = float(warmup)

        config._set_schedule_attrs(key, schedule_type, start_value, end_value, from_pos, to_pos)
        if warmup > 0.0:
            setattr(config, f"{key}_schedule_warmup", warmup)


def resolve_schedule_defaults(config) -> None:
    schedule_suffix = "_schedule"
    for key in list(vars(config).keys()):
        if not key.endswith(schedule_suffix):
            continue
        schedule = getattr(config, key)
        if not schedule:
            continue

        param = key[: -len(schedule_suffix)]
        config._default_schedule_attr(f"{param}_schedule_start_value", getattr(config, param))
        config._default_schedule_attr(f"{param}_schedule_end_value", 0.0)
        config._default_schedule_attr(f"{param}_schedule_start", 0.0)
        config._default_schedule_attr(f"{param}_schedule_end", 1.0)


def validate_schedules(config) -> None:
    schedule_suffix = "_schedule"
    for key in list(vars(config).keys()):
        if not key.endswith(schedule_suffix):
            continue

        schedule = getattr(config, key)
        if not schedule:
            continue

        param = key[: -len(schedule_suffix)]
        start_value = getattr(config, f"{param}_schedule_start_value", None)
        end_value = getattr(config, f"{param}_schedule_end_value", None)
        assert start_value is not None and end_value is not None, (
            f"{param}_schedule requires start and end values to be defined."
        )

        start_pos = getattr(config, f"{param}_schedule_start", None) or 0.0
        end_pos = getattr(config, f"{param}_schedule_end", None)
        if end_pos is None:
            assert config.max_env_steps is not None, (
                f"{param}_schedule requires max_env_steps or an explicit schedule_end value."
            )
            end_pos = 1.0

        assert start_pos >= 0 and end_pos >= 0, (
            f"{param}_schedule start/end must be non-negative."
        )
        assert end_pos > start_pos, (
            f"{param}_schedule end must be > start (degenerate schedule not allowed: "
            f"start={start_pos}, end={end_pos})."
        )
        assert not (config.max_env_steps is None and (start_pos <= 1.0 or end_pos <= 1.0)), (
            f"{param}_schedule uses fractional start/end positions but config.max_env_steps is not set."
        )
