"""Training launcher utilities.

Encapsulates train.py logic so the entrypoint stays minimal.
"""

from __future__ import annotations

import os
import sys
import wandb
from pathlib import Path
from dataclasses import asdict
from math import gcd
from typing import Optional

from agents import build_agent

from utils.user import prompt_confirm
from utils.config import Config
from utils.training_summary import present_prefit_summary
from utils.environment_registry import list_available_environments

def _parse_config_overrides(override_list):
    """Parse KEY=VALUE strings into dict of overrides.

    Handles type inference:
    - Numeric values: converted to int or float
    - Boolean values: 'true'/'false' (case-insensitive) → bool
    - Otherwise: kept as string
    """
    if not override_list:
        return {}

    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected KEY=VALUE")

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Type inference
        if value.lower() in ("true", "false"):
            overrides[key] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            # Numeric value
            overrides[key] = float(value) if "." in value else int(value)
        else:
            # String value
            overrides[key] = value

    return overrides

def _apply_config_overrides(config, overrides):
    """Apply dict of overrides to config object.

    Validates that each key exists as a config field before applying.
    """
    if not overrides:
        return config

    from dataclasses import fields
    valid_fields = {f.name for f in fields(config)}

    for key, value in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"Invalid config field: {key}. Not a valid Config attribute.")
        setattr(config, key, value)
        print(f"Override applied: {key} = {value}")

    return config

def _maybe_merge_wandb_config(config, *, wandb_sweep_flag: bool):
    """Optionally merge W&B sweep overrides into Config.

    Honors explicit --wandb_sweep or auto-detects via WANDB_SWEEP_ID/SWEEP_ID.
    Re-applies CLI --max-steps after merge to ensure CLI takes precedence.
    """

    # In case this is not a wandb sweep, do nothing (return original config)
    wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    is_wandb_sweep = bool(wandb_sweep_flag) or bool(wandb_sweep_id)
    if not is_wandb_sweep: return config

    # Generate a unique run ID or use pre-generated one from Modal
    # This ensures the W&B run name matches the ID (which becomes the local run dir name)
    from utils.formatting import sanitize_name
    project_name = config.project_id if config.project_id else sanitize_name(config.env_id)
    run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()

    # Initialize wandb with the original config (add algo_id since it's a property)
    config_dict = asdict(config)
    config_dict["algo_id"] = config.algo_id
    wandb.init(project=project_name, id=run_id, name=run_id, config=config_dict)

    # Merge sweep overrides back into the original config dict
    for key, value in dict(wandb.config).items():
        config_dict[key] = value

    # Build config from merged dict (preserves all required fields like algo_id)
    merged_config = Config.build_from_dict(config_dict)
    return merged_config

def _maybe_merge_debugger_config(config):
    """When a debugger is attached, force single-env, synchronous vectorization.

    Rationale: debuggers and async vec envs don't play nicely. To keep
    breakpoints usable, clamp to `n_envs=1` and `vectorization_mode='sync'`.
    Also adjust `batch_size` to remain a clean divisor of the rollout size so
    training proceeds without violating config invariants.
    """
    # Detect common debuggers (e.g., pdb, debugpy) via active trace function
    if sys.gettrace() is None:
        return config

    # Preserve original values for logging and ratio-based batch recompute
    orig_n_envs = int(getattr(config, "n_envs", 1) or 1)
    orig_vectorization_mode = getattr(config, "vectorization_mode", "auto")
    n_steps = int(getattr(config, "n_steps", 1) or 1)
    orig_batch = int(getattr(config, "batch_size", 1) or 1)

    # Compute original rollout size and batch ratio (batch per rollout)
    orig_rollout = max(1, int(orig_n_envs) * int(n_steps))
    ratio = float(orig_batch) / float(orig_rollout) if orig_rollout > 0 else 1.0

    # Apply debugger-safe settings
    config.n_envs = 1
    config.vectorization_mode = "sync"

    # Recompute batch_size to fit the new rollout while preserving ratio when possible
    new_rollout = max(1, int(config.n_envs) * int(n_steps))
    new_batch = max(1, int(new_rollout * ratio))
    if new_batch > new_rollout:
        new_batch = new_rollout
    if new_rollout % new_batch != 0:
        d = gcd(int(new_rollout), int(new_batch))
        new_batch = int(d) if int(d) > 0 else 1
    old_batch = getattr(config, "batch_size", new_batch)
    config.batch_size = int(new_batch)

    print(
        f"Debugger detected: forcing n_envs=1, vectorization_mode='sync'; "
        f"batch_size {old_batch}→{config.batch_size} (was n_envs={orig_n_envs}, vectorization_mode='{orig_vectorization_mode}')."
    )
    return config

# TODO: remove if exception not raised
def _ensure_wandb_run_initialized(config) -> None:
    """Ensure a W&B run exists before agent construction.

    - If running under a W&B Sweep, `_maybe_merge_wandb_config` already called
      `wandb.init`, so this becomes a no-op.
    - Otherwise, initialize a run with the project's name and full config.
    - If WANDB_RUN_ID env var is set (e.g., from Modal training), use that ID.
    """

    # If W&B is disabled, do nothing
    if not getattr(config, 'enable_wandb', True): return

    # If a run is already active (e.g., sweep agent), do nothing
    if wandb.run is not None: return

    # Otherwise create a fresh run using project and full config
    # Generate a unique run ID or use pre-generated one from Modal
    # This ensures the W&B run name matches the ID (which becomes the local run dir name)
    project_name = config.project_id
    assert project_name, "project_id is required"
    run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb.init(project=project_name, id=run_id, name=run_id, config=asdict(config))

def _extract_elapsed_seconds(agent) -> Optional[float]:
    """Return elapsed seconds from agent without broad exception handling.

    Prefers an explicit cached value, falling back to timings tracker when
    available. Returns None when not determinable.
    """
    cached = getattr(agent, "_fit_elapsed_seconds", None)
    if cached is not None:
        try:
            return float(cached)
        except (TypeError, ValueError):
            return None

    timings = getattr(agent, "timings", None)
    if timings is None or not hasattr(timings, "seconds_since"):
        return None
    try:
        return float(timings.seconds_since("on_fit_start"))
    except (TypeError, ValueError, KeyError, RuntimeError):
        return None


def _launch_training_resume(args) -> None:
    """Launch training in resume mode from an existing checkpoint."""
    from utils.config import Config
    from utils.formatting import format_duration
    from utils.random import set_random_seed
    from utils.run import Run
    from utils.io import read_json
    import wandb

    # Resolve run ID (handle @last symlink)
    run_id = args.resume
    if run_id == "@last":
        from utils.run import LAST_RUN_DIR
        if not LAST_RUN_DIR.exists():
            raise FileNotFoundError("No @last run found. Train a model first.")
        run_id = LAST_RUN_DIR.resolve().name

    # Check if run exists locally, if not try to download from W&B
    run_dir = Run._resolve_run_dir(run_id)
    if not run_dir.exists():
        print(f"Run {run_id} not found locally. Attempting to download from W&B...")
        from utils.wandb_artifacts import download_run_artifact
        download_run_artifact(run_id)

    # Load run
    print(f"Resuming run: {run_id}")
    run = Run.load(run_id)

    # Resolve checkpoint directory
    checkpoint_dir = _resolve_checkpoint_dir(run, args.epoch)
    print(f"Loading checkpoint from: {checkpoint_dir}")

    # Load config from checkpoint (prefer state.json, fallback to run's config.json)
    state_path = checkpoint_dir / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        config_dict = state.get("config")
        if config_dict and "algo_id" in config_dict:
            config = Config.build_from_dict(config_dict)
        else:
            # Old checkpoint format, fallback to run's config.json
            print("Warning: Checkpoint uses old format, loading config from run directory")
            config = run.load_config()
            state = None
    else:
        # Very old checkpoint format without state.json
        print("Warning: Checkpoint missing state.json, loading config from run directory")
        config = run.load_config()
        state = None

    # Apply CLI overrides (generic --override flags)
    if hasattr(args, 'overrides') and args.overrides:
        overrides_dict = _parse_config_overrides(args.overrides)
        config = _apply_config_overrides(config, overrides_dict)

    # Allow overriding max_env_steps from CLI (takes precedence over --override)
    cli_max_env_steps = int(args.max_env_steps) if args.max_env_steps else None
    if cli_max_env_steps is not None:
        print(f"Overriding max_env_steps: {config.max_env_steps} → {cli_max_env_steps}")
        config.max_env_steps = cli_max_env_steps

    # Initialize W&B if enabled
    if getattr(config, 'enable_wandb', True):
        # Resume existing W&B run
        # Use run_id as both id and name (name should match existing run, this is just explicit)
        from utils.formatting import sanitize_name
        project_name = config.project_id if config.project_id else sanitize_name(config.env_id)
        from dataclasses import asdict
        wandb.init(
            project=project_name,
            id=run_id,
            name=run_id,
            resume="must",
            config=asdict(config)
        )

    # Set random seed (will be overridden by checkpoint RNG states)
    set_random_seed(config.seed)

    # Build agent
    from agents import build_agent
    agent = build_agent(config)

    # Attach run to agent (reuse existing run, don't create new one)
    agent.run = run

    # Load checkpoint into agent
    agent.load_checkpoint(checkpoint_dir, resume_training=True)

    # Set the epoch to resume from (trainer will continue from this epoch)
    if state:
        loaded_epoch = state.get("epoch", 0)
        agent._resume_from_epoch = loaded_epoch
        print(f"Continuing training from epoch {loaded_epoch}")
    else:
        print("Warning: Cannot determine checkpoint epoch, starting from 0")

    # Train (trainer will start from loaded epoch)
    agent.learn()

    # Print completion message
    elapsed_seconds = _extract_elapsed_seconds(agent)
    human = format_duration(elapsed_seconds) if isinstance(elapsed_seconds, (int, float)) else "unknown"
    reason = (
        getattr(agent, "_final_stop_reason", None)
        or getattr(agent, "_early_stop_reason", None)
        or "completed."
    )
    if isinstance(reason, str) and reason and not str(reason).endswith(".") and reason != "completed.":
        reason = f"{reason}."
    print(f"Training completed in {human}. Reason: {reason}")


def _load_pretrained_weights(agent, run_spec: str) -> None:
    """Load pretrained weights from another run's checkpoint.

    Args:
        agent: Agent to load weights into
        run_spec: Run specification in format "run_id" or "run_id/checkpoint"
                 Examples:
                 - "abc123" -> use @best if available, else @last
                 - "abc123/@best" -> explicitly use @best
                 - "abc123/@last" -> explicitly use @last
                 - "abc123/epoch=13" -> use epoch 13
                 - "@last" -> most recent run, @best if available else @last
                 - "@last/@best" -> most recent run, @best checkpoint
    """
    from utils.run import Run, LAST_RUN_DIR

    # Parse run_spec into run_id and checkpoint_spec
    if "/" in run_spec:
        run_id, checkpoint_spec = run_spec.split("/", 1)
    else:
        run_id = run_spec
        checkpoint_spec = None

    # Resolve run ID (handle @last symlink)
    if run_id == "@last":
        if not LAST_RUN_DIR.exists():
            raise FileNotFoundError("No @last run found. Train a model first.")
        run_id = LAST_RUN_DIR.resolve().name

    # Check if run exists locally, if not try to download from W&B
    run_dir = Run._resolve_run_dir(run_id)
    if not run_dir.exists():
        print(f"Run {run_id} not found locally. Attempting to download from W&B...")
        from utils.wandb_artifacts import download_run_artifact
        download_run_artifact(run_id)

    # Load run
    print(f"Loading pretrained weights from run: {run_id}")
    run = Run.load(run_id)

    # Resolve checkpoint directory
    checkpoint_dir = _resolve_checkpoint_dir(run, epoch_spec=checkpoint_spec)
    checkpoint_desc = checkpoint_spec if checkpoint_spec else "(@best if available, else @last)"
    print(f"Loading weights from: {checkpoint_dir} {checkpoint_desc}")

    # Load only model weights (not optimizer/RNG states)
    # Use strict=False to allow partial loading for transfer learning across different architectures
    agent.load_checkpoint(checkpoint_dir, resume_training=False, strict=False)
    print(f"Pretrained weights loaded successfully from {run_id}")


def _resolve_checkpoint_dir(run, epoch_spec: Optional[str]) -> Path:
    """Resolve checkpoint directory from epoch spec.

    Args:
        run: Run object
        epoch_spec: Epoch specifier: None, '@best', '@last', 'epoch=N', or epoch number

    Returns:
        Path to checkpoint directory
    """
    from pathlib import Path

    # Default: prefer @best if exists, else @last
    if epoch_spec is None:
        if run.best_checkpoint_dir.exists():
            return run.best_checkpoint_dir
        elif run.last_checkpoint_dir.exists():
            return run.last_checkpoint_dir
        else:
            raise FileNotFoundError(f"No checkpoints found for run {run.run_id}")

    # Handle symlinks
    if epoch_spec == "@best":
        if not run.best_checkpoint_dir.exists():
            raise FileNotFoundError(f"No best checkpoint found for run {run.run_id}")
        return run.best_checkpoint_dir
    elif epoch_spec == "@last":
        if not run.last_checkpoint_dir.exists():
            raise FileNotFoundError(f"No last checkpoint found for run {run.run_id}")
        return run.last_checkpoint_dir

    # Handle specific epoch number (supports both "13" and "epoch=13" formats)
    epoch_str = epoch_spec
    if epoch_spec.startswith("epoch="):
        epoch_str = epoch_spec[6:]  # Strip "epoch=" prefix

    try:
        epoch = int(epoch_str)
        checkpoint_dir = run.checkpoint_dir_for_epoch(epoch)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found in run {run.run_id}")
        return checkpoint_dir
    except ValueError:
        raise ValueError(f"Invalid epoch spec: {epoch_spec}. Use '@best', '@last', 'epoch=N', or an integer.")


def launch_training_from_args(args) -> None:
    """End-to-end training launcher extracted from train.py.

    Keeps the entrypoint focused on argument parsing while this
    function handles config resolution, seeding, agent setup, and
    post-training reporting.
    """
    from utils.config import Config, load_config
    from utils.formatting import format_duration
    from utils.random import set_random_seed
    from utils.run import Run
    from utils.wandb_workspace import create_or_update_workspace_for_current_run

    # Handle resume mode
    if args.resume:
        _launch_training_resume(args)
        return

    # Resolve configuration spec from positional, then flag, then default
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec: raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    env_id, variant_id = config_spec.split(":", 1)

    # Load the requested configuration
    config = load_config(env_id, variant_id)

    # Disable W&B if WANDB_MODE is set to disabled
    import os
    if os.environ.get("WANDB_MODE") == "disabled":
        config.enable_wandb = False

    # Resolve policy type early (auto-select CNN for image observations)
    from utils.policy_factory import resolve_policy_type_for_config
    resolve_policy_type_for_config(config)

    present_prefit_summary(config)

    # Detect if we're in sweep mode
    wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    is_wandb_sweep = bool(args.wandb_sweep) or bool(wandb_sweep_id)

    # Prompt if user wants to start training (skip prompt in sweep mode)
    start_training = prompt_confirm("Start training?", default=True, quiet=is_wandb_sweep)
    if not start_training:
        print("Training aborted before initialization.")
        return

    # In case this is a wandb sweep, merge the sweep config
    config = _maybe_merge_wandb_config(config, wandb_sweep_flag=args.wandb_sweep)

    # When running with a debugger, force single-env,
    # in-process execution (easier to debug)
    config = _maybe_merge_debugger_config(config)

    # Apply CLI overrides (generic --override flags)
    if hasattr(args, 'overrides') and args.overrides:
        overrides_dict = _parse_config_overrides(args.overrides)
        config = _apply_config_overrides(config, overrides_dict)

    # Override max env steps if provided through CLI (takes precedence over --override)
    cli_max_env_steps = int(args.max_env_steps) if args.max_env_steps else None
    if cli_max_env_steps is not None: config.max_env_steps = cli_max_env_steps

    # Ensure a W&B run exists (sweep-or-regular) before building the agent
    _ensure_wandb_run_initialized(config)

    # Create/update the W&B workspace immediately so the dashboard is ready during training
    # Skip workspace creation during sweeps (not needed and may cause issues)
    if getattr(config, 'enable_wandb', True) and not is_wandb_sweep:
        create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True) # TODO: review this function

    # Set global RNG seed for reproducibility
    set_random_seed(config.seed)

    # Create the agent and kick off learning
    agent = build_agent(config)

    # Load pretrained weights if requested (CLI arg takes precedence over config)
    init_from_run = (hasattr(args, 'init_from_run') and args.init_from_run) or config.init_from_run
    if init_from_run:
        _load_pretrained_weights(agent, init_from_run)

    agent.learn()

    # Print final training completion message (duration and normalized reason)
    elapsed_seconds = _extract_elapsed_seconds(agent)
    human = format_duration(elapsed_seconds) if isinstance(elapsed_seconds, (int, float)) else "unknown"
    reason = (
        getattr(agent, "_final_stop_reason", None)
        or getattr(agent, "_early_stop_reason", None)
        or "completed."
    )
    if isinstance(reason, str) and reason and not str(reason).endswith(".") and reason != "completed.":
        reason = f"{reason}."
    print(f"Training completed in {human}. Reason: {reason}")
