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

from utils.logging import display_config_summary
from utils.user import prompt_confirm
from utils.config import Config

def _maybe_merge_wandb_config(config, *, wandb_sweep_flag: bool):
    """Optionally merge W&B sweep overrides into Config.

    Honors explicit --wandb_sweep or auto-detects via WANDB_SWEEP_ID/SWEEP_ID.
    Re-applies CLI --max-steps after merge to ensure CLI takes precedence.
    """

    # In case this is not a wandb sweep, do nothing (return original config)
    wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    is_wandb_sweep = bool(wandb_sweep_flag) or bool(wandb_sweep_id)
    if not is_wandb_sweep: return config

    # TODO: confirm this is working as expected
    # Otherwise, merge the configs and return
    wandb.init(config=asdict(config))
    merged = dict(wandb.config)
    merged_config = Config.build_from_dict(merged)
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
    """

    # If W&B is disabled, do nothing
    if not getattr(config, 'enable_wandb', True): return

    # If a run is already active (e.g., sweep agent), do nothing
    if wandb.run is not None: return

    # Otherwise create a fresh run using project and full config
    from utils.formatting import sanitize_name
    project_name = sanitize_name(config.env_id)
    wandb.init(project=project_name, config=asdict(config))

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


def _format_summary_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    return value


def _present_prefit_summary(config) -> None:
    spec = config.spec if isinstance(getattr(config, "spec", None), dict) else {}
    returns = spec.get("returns") if isinstance(spec, dict) else {}
    rewards = spec.get("rewards") if isinstance(spec, dict) else {}

    reward_threshold = None
    if isinstance(returns, dict) and returns.get("threshold_solved") is not None:
        reward_threshold = returns.get("threshold_solved")
    elif isinstance(rewards, dict) and rewards.get("threshold_solved") is not None:
        reward_threshold = rewards.get("threshold_solved")

    env_block = {
        "env_id": _format_summary_value(config.env_id),
        "obs_type": _format_summary_value(config.obs_type),
        "wrappers": _format_summary_value(config.env_wrappers),
        "vectorization_mode": _format_summary_value(config.vectorization_mode),
        "frame_stack": _format_summary_value(getattr(config, "frame_stack", None)),
        "normalize_obs": _format_summary_value(getattr(config, "normalize_obs", None)),
        "normalize_reward": _format_summary_value(getattr(config, "normalize_reward", None)),
        "grayscale_obs": _format_summary_value(getattr(config, "grayscale_obs", None)),
        "resize_obs": _format_summary_value(getattr(config, "resize_obs", None)),
        "spec/action_space": _format_summary_value(spec.get("action_space")),
        "spec/observation_space": _format_summary_value(spec.get("observation_space")),
        "reward_threshold": _format_summary_value(reward_threshold),
        "time_limit": _format_summary_value(spec.get("max_episode_steps")),
    }

    training_block = {
        "algo_id": _format_summary_value(getattr(config, "algo_id", None)),
        "policy": _format_summary_value(getattr(config, "policy", None)),
        "hidden_dims": _format_summary_value(getattr(config, "hidden_dims", None)),
        "activation": _format_summary_value(getattr(config, "activation", None)),
        "optimizer": _format_summary_value(getattr(config, "optimizer", None)),
        "seed": _format_summary_value(getattr(config, "seed", None)),
        "n_envs": _format_summary_value(getattr(config, "n_envs", None)),
        "n_steps": _format_summary_value(getattr(config, "n_steps", None)),
        "n_epochs": _format_summary_value(getattr(config, "n_epochs", None)),
        "batch_size": _format_summary_value(getattr(config, "batch_size", None)),
        "max_env_steps": _format_summary_value(getattr(config, "max_env_steps", None)),
        "policy_lr": _format_summary_value(getattr(config, "policy_lr", None)),
        "gamma": _format_summary_value(getattr(config, "gamma", None)),
        "gae_lambda": _format_summary_value(getattr(config, "gae_lambda", None)),
        "ent_coef": _format_summary_value(getattr(config, "ent_coef", None)),
        "vf_coef": _format_summary_value(getattr(config, "vf_coef", None)),
        "clip_range": _format_summary_value(getattr(config, "clip_range", None)),
        "max_grad_norm": _format_summary_value(getattr(config, "max_grad_norm", None)),
        "returns_type": _format_summary_value(getattr(config, "returns_type", None)),
        "advantages_type": _format_summary_value(getattr(config, "advantages_type", None)),
    }

    project_id = getattr(config, "project_id", None) or getattr(config, "env_id", None)
    run_block = {
        "project_id": _format_summary_value(project_id),
        "run_id": "<pending>",
        "quiet": _format_summary_value(getattr(config, "quiet", False)),
    }

    display_config_summary({
        "Run": run_block,
        "Environment": env_block,
        "Training": training_block,
    })


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

    # Allow overriding max_env_steps from CLI
    cli_max_env_steps = int(args.max_env_steps) if args.max_env_steps else None
    if cli_max_env_steps is not None:
        print(f"Overriding max_env_steps: {config.max_env_steps} → {cli_max_env_steps}")
        config.max_env_steps = cli_max_env_steps

    # Initialize W&B if enabled
    if getattr(config, 'enable_wandb', True):
        # Resume existing W&B run
        from utils.formatting import sanitize_name
        project_name = config.project_id if config.project_id else sanitize_name(config.env_id)
        from dataclasses import asdict
        wandb.init(
            project=project_name,
            id=run_id,
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


def _resolve_checkpoint_dir(run, epoch_spec: Optional[str]) -> Path:
    """Resolve checkpoint directory from epoch spec.

    Args:
        run: Run object
        epoch_spec: Epoch specifier: None, '@best', '@last', or epoch number

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

    # Handle specific epoch number
    try:
        epoch = int(epoch_spec)
        checkpoint_dir = run.checkpoint_dir_for_epoch(epoch)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found in run {run.run_id}")
        return checkpoint_dir
    except ValueError:
        raise ValueError(f"Invalid epoch spec: {epoch_spec}. Use '@best', '@last', or an integer.")


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

    _present_prefit_summary(config)

    # Prompt if user wants to start training
    start_training = prompt_confirm("Start training?", default=True)
    if not start_training:
        print("Training aborted before initialization.")
        return

    # In case this is a wandb sweep, merge the sweep config
    config = _maybe_merge_wandb_config(config, wandb_sweep_flag=args.wandb_sweep)

    # When running with a debugger, force single-env, 
    # in-process execution (easier to debug)
    config = _maybe_merge_debugger_config(config)

    # Override max env steps if provided through CLI
    cli_max_env_steps = int(args.max_env_steps) if args.max_env_steps else None
    if cli_max_env_steps is not None: config.max_env_steps = cli_max_env_steps

    # Ensure a W&B run exists (sweep-or-regular) before building the agent
    _ensure_wandb_run_initialized(config)

    # Create/update the W&B workspace immediately so the dashboard is ready during training
    if getattr(config, 'enable_wandb', True):
        create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True) # TODO: review this function

    # Set global RNG seed for reproducibility
    set_random_seed(config.seed)

    # Create the agent and kick off learning
    agent = build_agent(config)
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

def find_closest_match(search_term, candidates):
    """Find the closest match for a search term among candidates using fuzzy matching."""
    if not search_term:
        return None
    
    search_lower = search_term.lower()
    candidates_lower = [c.lower() for c in candidates]
    
    # Exact match first
    for i, candidate in enumerate(candidates_lower):
        if search_lower == candidate:
            return candidates[i]
    
    # Substring match
    for i, candidate in enumerate(candidates_lower):
        if search_lower in candidate or candidate in search_lower:
            return candidates[i]
    
    # Word-based matching (split on hyphens and underscores)
    search_words = set(search_lower.replace('-', ' ').replace('_', ' ').split())
    
    best_match = None
    best_score = 0
    
    for i, candidate in enumerate(candidates_lower):
        candidate_words = set(candidate.replace('-', ' ').replace('_', ' ').split())
        
        # Calculate overlap score
        overlap = len(search_words.intersection(candidate_words))
        if overlap > best_score:
            best_score = overlap
            best_match = candidates[i]
    
    return best_match if best_score > 0 else None


# TODO: create environment registry class instead
def list_available_environments(search_term=None, exact_match=None):
    """List all available environment targets with their descriptions."""
    from utils.config import Config
    from utils.io import read_yaml
    
    # ANSI escape codes for styling
    BOLD = '\033[1m'
    RESET = '\033[0m'
    BULLET = '•'
    
    # Assert that config/environments directory exists
    config_dir = Path("config/environments")
    if not config_dir.exists(): raise FileNotFoundError("config/environments directory not found")
    
    # List all env names
    yaml_files = sorted(config_dir.glob("*.yaml"))
    env_names = [f.stem for f in yaml_files]
    
    # If exact_match provided, use it directly
    if exact_match:
        yaml_files = [f for f in yaml_files if f.stem == exact_match]
        if not yaml_files:
            print(f"Environment '{exact_match}' not found.")
            return
        print(f"{BOLD}Environment targets for '{exact_match}':{RESET}")
    # If search term provided, find closest match
    elif search_term:
        matched_env = find_closest_match(search_term, env_names)
        if not matched_env:
            print(f"No environment found matching '{search_term}'")
            print(f"Available environments: {', '.join(env_names)}")
            return
        
        # Filter to only the matched environment
        yaml_files = [f for f in yaml_files if f.stem == matched_env]
        print(f"{BOLD}Environment targets for '{matched_env}':{RESET}")
    else:
        print(f"{BOLD}Available Environment Targets:{RESET}")
    
    print()
    
    for yaml_file in yaml_files:
        # Load the YAML file
        doc = read_yaml(yaml_file) or {}
        
        # Get the environment name from the filename
        env_name = yaml_file.stem
        
        # Find all public targets (non-underscore keys that are dictionaries)
        config_field_names = set(Config.__dataclass_fields__.keys())
        public_targets = []
        
        for key, value in doc.items():
            # Skip base config fields and non-dict fields
            if key in config_field_names or not isinstance(value, dict):
                continue
                
            # Skip meta/utility sections (e.g., anchors) prefixed with underscore
            if isinstance(key, str) and key.startswith("_"):
                continue
                
            # This is a public target
            description = value.get("description", "No description available")
            public_targets.append((key, description))
        
        if public_targets:
            # Use bold formatting for environment name
            print(f"{BOLD}{env_name}:{RESET}")
            for target, description in sorted(public_targets):
                print(f"  {BULLET} {env_name}:{target} - {description}")
            print()
