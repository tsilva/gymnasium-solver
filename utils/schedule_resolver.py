"""Helper for resolving schedule configuration from config into callback instances."""

from typing import Callable, Optional

from trainer_callbacks.hyperparameter_scheduler import HyperparameterSchedulerCallback


def schedule_pos_to_vec_steps(
    raw: Optional[float],
    *,
    param: str,
    default_to_max: bool,
    max_env_steps: Optional[int],
    n_envs: int,
) -> float:
    """Convert schedule position (fraction or env_steps) to vec_steps.

    Args:
        raw: Schedule position as fraction (0-1) or absolute env_steps (>1)
        param: Parameter name for error messages
        default_to_max: If True and raw is None, default to max_env_steps
        max_env_steps: Maximum environment steps for training
        n_envs: Number of parallel environments

    Returns:
        Position in vec_steps
    """
    if raw is None:
        if default_to_max:
            if max_env_steps is None:
                raise ValueError(
                    f"{param}_schedule requires config.max_env_steps or an explicit {param}_schedule_end."
                )
            # max_env_steps is in env_steps, convert to vec_steps
            return float(max_env_steps) / n_envs
        return 0.0

    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{param}_schedule start/end must be non-negative.")

    # Fractional position: interpret as fraction of max_env_steps
    if value <= 1.0:
        if max_env_steps is None:
            raise ValueError(
                f"{param}_schedule uses fractional start/end but config.max_env_steps is not set."
            )
        env_steps = value * float(max_env_steps)
        return env_steps / n_envs

    # Absolute position: interpret as env_steps, convert to vec_steps
    return value / n_envs


def build_hyperparameter_scheduler_callbacks(
    config,
    module,
    set_value_fn_map: Optional[dict[str, Callable]] = None,
) -> list[HyperparameterSchedulerCallback]:
    """Build hyperparameter scheduler callbacks from config.

    Scans config for fields ending with '_schedule' and creates scheduler callbacks.

    Args:
        config: Config object with schedule fields
        module: Module instance that will receive hyperparameter updates
        set_value_fn_map: Optional mapping of param names to custom setter functions

    Returns:
        List of HyperparameterSchedulerCallback instances
    """
    callbacks = []
    schedule_suffix = "_schedule"
    max_env_steps = config.max_env_steps
    n_envs = config.n_envs
    set_value_fn_map = set_value_fn_map or {}

    for key, value in vars(config).items():
        if not key.endswith(schedule_suffix) or not value:
            continue

        param = key[: -len(schedule_suffix)]
        assert hasattr(module, param), f"Module {module} has no attribute {param}"

        start_value = getattr(config, f"{param}_schedule_start_value", None)
        end_value = getattr(config, f"{param}_schedule_end_value", None)
        if start_value is None or end_value is None:
            raise ValueError(f"{param}_schedule requires start/end values in the config.")

        start_pos_raw = getattr(config, f"{param}_schedule_start", None)
        end_pos_raw = getattr(config, f"{param}_schedule_end", None)

        start_step = schedule_pos_to_vec_steps(
            start_pos_raw,
            param=param,
            default_to_max=False,
            max_env_steps=max_env_steps,
            n_envs=n_envs,
        )
        end_step = schedule_pos_to_vec_steps(
            end_pos_raw,
            param=param,
            default_to_max=True,
            max_env_steps=max_env_steps,
            n_envs=n_envs,
        )
        warmup_fraction = getattr(config, f"{param}_schedule_warmup", 0.0)

        callbacks.append(
            HyperparameterSchedulerCallback(
                schedule=value,
                parameter=param,
                start_value=float(start_value),
                end_value=float(end_value),
                start_step=float(start_step),
                end_step=float(end_step),
                warmup_fraction=float(warmup_fraction),
                set_value_fn=set_value_fn_map.get(param, None),
            )
        )

    return callbacks
