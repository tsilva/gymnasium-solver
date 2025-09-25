"""Training launcher utilities.

Encapsulates train.py logic so the entrypoint stays minimal.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import asdict
from math import gcd
from typing import Optional

from utils.logging import display_config_summary
from utils.user import prompt_confirm


def _parse_positive_int(value: str, flag: str) -> int:
    """Parse a positive integer from CLI strings (supports 1e5 and 1_000 forms).

    Raises SystemExit with a friendly message on invalid input to match
    typical CLI behavior in train.py.
    """
    sanitized = str(value).replace("_", "")
    try:
        numeric = float(sanitized)
    except ValueError as exc:
        raise SystemExit(f"{flag} must be a positive integer (got '{value}').") from exc
    if numeric <= 0:
        raise SystemExit(f"{flag} must be greater than zero (got {value}).")
    if not numeric.is_integer():
        raise SystemExit(f"{flag} must be a whole number (got {value}).")
    return int(numeric)


def _init_wandb_sweep(config):
    """Initialize W&B early for sweeps and merge wandb.config into config.

    Returns a Config instance with overrides applied.
    """
    # Import locally to keep module import light for tests and non-W&B flows
    import wandb
    from utils.config import Config

    base = asdict(config)
    wandb.init(config=base)
    merged = dict(wandb.config)
    return Config.build_from_dict(merged)


def _maybe_merge_wandb_config(config, *, wandb_sweep_flag: bool, cli_max_timesteps: Optional[int]):
    """Optionally merge W&B sweep overrides into Config.

    Honors explicit --wandb_sweep or auto-detects via WANDB_SWEEP_ID/SWEEP_ID.
    Re-applies CLI --max-steps after merge to ensure CLI takes precedence.
    """
    use_wandb_sweep = bool(wandb_sweep_flag) or bool(
        os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    )
    if not use_wandb_sweep:
        return config

    config = _init_wandb_sweep(config)
    if cli_max_timesteps is not None:
        config.max_timesteps = cli_max_timesteps
    return config


def _apply_debugger_env_overrides(config):
    """When a debugger is attached, force single-env, in-process execution.

    Rationale: debuggers and subprocess-based vec envs don't play nicely. To
    keep breakpoints usable, clamp to `n_envs=1` and `subproc=False`. Also
    adjust `batch_size` to remain a clean divisor of the rollout size so
    training proceeds without violating config invariants.
    """
    # Detect common debuggers (e.g., pdb, debugpy) via active trace function
    if sys.gettrace() is None:
        return config

    # Preserve original values for logging and ratio-based batch recompute
    orig_n_envs = int(getattr(config, "n_envs", 1) or 1)
    orig_subproc = getattr(config, "subproc", None)
    n_steps = int(getattr(config, "n_steps", 1) or 1)
    orig_batch = int(getattr(config, "batch_size", 1) or 1)

    # Compute original rollout size and batch ratio (batch per rollout)
    orig_rollout = max(1, int(orig_n_envs) * int(n_steps))
    ratio = float(orig_batch) / float(orig_rollout) if orig_rollout > 0 else 1.0

    # Apply debugger-safe settings
    config.n_envs = 1
    config.subproc = False

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
        f"Debugger detected: forcing n_envs=1, subproc=False; "
        f"batch_size {old_batch}→{config.batch_size} (was n_envs={orig_n_envs}, subproc={orig_subproc})."
    )
    return config


def _ensure_wandb_run_initialized(config) -> None:
    """Ensure a W&B run exists before agent construction.

    - If running under a W&B Sweep, `_maybe_merge_wandb_config` already called
      `wandb.init`, so this becomes a no-op.
    - Otherwise, initialize a run with the project's name and full config.
    """
    import wandb  # lazy import to keep non-W&B paths light

    # If a run is already active (e.g., sweep agent), do nothing
    if wandb.run is not None: return

    # Otherwise create a fresh run using project and full config
    from utils.formatting import sanitize_name
    raw_project = getattr(config, "project_id", None) or getattr(config, "env_id", None) or "gymnasium-solver"
    project_name = sanitize_name(raw_project)
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
        "env_id": _format_summary_value(getattr(config, "env_id", None)),
        "obs_type": _format_summary_value(getattr(config, "obs_type", None)),
        "wrappers": _format_summary_value(getattr(config, "env_wrappers", None)),
        "subproc": _format_summary_value(getattr(config, "subproc", None)),
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
        "seed": _format_summary_value(getattr(config, "seed", None)),
        "n_envs": _format_summary_value(getattr(config, "n_envs", None)),
        "n_steps": _format_summary_value(getattr(config, "n_steps", None)),
        "batch_size": _format_summary_value(getattr(config, "batch_size", None)),
        "max_timesteps": _format_summary_value(getattr(config, "max_timesteps", None)),
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


def launch_training_from_args(args) -> None:
    """End-to-end training launcher extracted from train.py.

    Keeps the entrypoint focused on argument parsing while this
    function handles config resolution, seeding, agent setup, and
    post-training reporting.
    """
    from stable_baselines3.common.utils import set_random_seed

    from utils.config import load_config
    from utils.formatting import format_duration
    from utils.wandb_workspace import create_or_update_workspace_for_current_run

    # Resolve configuration spec from positional, then flag, then default
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec:
        raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    env_id, variant_id = config_spec.split(":", 1)

    # Parse CLI max-steps override early (after arg parsing) so we can apply it
    cli_max_timesteps = None
    if getattr(args, "max_timesteps", None) is not None:
        cli_max_timesteps = _parse_positive_int(str(args.max_timesteps), "--max-steps")

    # Load configuration and apply simple CLI overrides
    config = load_config(env_id, variant_id)
    if getattr(args, "quiet", False) is True:
        config.quiet = True
    if cli_max_timesteps is not None:
        config.max_timesteps = cli_max_timesteps

    _present_prefit_summary(config)

    quiet = bool(getattr(config, "quiet", False))
    start_training = prompt_confirm("Start training?", default=True, quiet=quiet)
    if not start_training:
        print("Training aborted before initialization.")
        return

    # Merge W&B sweep overrides when requested/auto-detected (after confirmation)
    prev_config = config
    config = _maybe_merge_wandb_config(
        config, wandb_sweep_flag=bool(getattr(args, "wandb_sweep", False)), cli_max_timesteps=cli_max_timesteps
    )
    if config is not prev_config:
        print("Applied W&B sweep overrides after confirmation.")

    # Reapply CLI/noise overrides that must persist after sweep merge
    if getattr(args, "quiet", False) is True:
        config.quiet = True
    if cli_max_timesteps is not None:
        config.max_timesteps = cli_max_timesteps

    # If running under a debugger (e.g., vscode/pycharm/debugpy), clamp vec envs
    config = _apply_debugger_env_overrides(config)

    # Set global RNG seed for reproducibility
    set_random_seed(config.seed)

    # Ensure a W&B run exists (sweep-or-regular) before building the agent
    _ensure_wandb_run_initialized(config)

    # Create/update the W&B workspace immediately so the dashboard is ready during training
    create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)

    # Create the agent and kick off learning
    from agents import build_agent

    agent = build_agent(config)
    setattr(agent, "_prefit_prompt_completed", True)
    maybe_warn = getattr(agent, "_maybe_warn_observation_policy_mismatch", None)
    if callable(maybe_warn):
        try:
            maybe_warn()
        except Exception:
            pass
    agent.learn()

    # (Workspace already ensured at training start)

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


def find_matching_environment(env_name):
    """Find the best matching environment for a given name."""
    from utils.config import Config
    from utils.io import read_yaml
    
    config_dir = Path("config/environments")
    if not config_dir.exists():
        return None
    
    # Check for exact match first
    env_file = config_dir / f"{env_name}.yaml"
    if env_file.exists():
        return env_name
    
    # Check for fuzzy match
    yaml_files = list(config_dir.glob("*.yaml"))
    env_names = [f.stem for f in yaml_files]
    matched_env = find_closest_match(env_name, env_names)
    
    return matched_env
