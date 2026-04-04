"""Training launcher utilities.

Encapsulates train.py logic so the entrypoint stays minimal.
"""

from __future__ import annotations

import os

from agents import build_agent

from utils.config import load_config
from utils.train_debugger import maybe_merge_debugger_config as _maybe_merge_debugger_config
from utils.train_overrides import (
    apply_cli_overrides,
    apply_config_overrides as _apply_config_overrides,
    apply_env_kwargs_overrides as _apply_env_kwargs_overrides,
    parse_config_overrides as _parse_config_overrides,
)
from utils.train_reporting import (
    extract_elapsed_seconds as _extract_elapsed_seconds,
    print_training_completion,
)
from utils.train_resume import (
    launch_training_resume as _launch_training_resume,
    load_pretrained_weights as _load_pretrained_weights,
)
from utils.train_wandb import (
    ensure_wandb_run_initialized as _ensure_wandb_run_initialized,
    maybe_merge_wandb_config as _maybe_merge_wandb_config,
)
from utils.training_summary import present_prefit_summary
from utils.user import prompt_confirm


def launch_training_from_args(args) -> None:
    """Resolve config, apply runtime overrides, and launch training."""
    from utils.policy_factory import resolve_policy_type_for_config
    from utils.random import set_random_seed
    from utils.wandb_workspace import create_or_update_workspace_for_current_run

    if args.resume:
        _launch_training_resume(args)
        return

    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec:
        raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    env_id, variant_id = config_spec.split(":", 1)

    config = load_config(env_id, variant_id)

    if os.environ.get("WANDB_MODE") == "disabled":
        config.enable_wandb = False

    resolve_policy_type_for_config(config)
    present_prefit_summary(config)

    wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    is_wandb_sweep = bool(args.wandb_sweep) or bool(wandb_sweep_id)

    start_training = prompt_confirm("Start training?", default=True, quiet=is_wandb_sweep)
    if not start_training:
        print("Training aborted before initialization.")
        return

    config = _maybe_merge_wandb_config(config, wandb_sweep_flag=args.wandb_sweep)
    config = _maybe_merge_debugger_config(config)
    config = apply_cli_overrides(config, args)

    _ensure_wandb_run_initialized(config)

    if getattr(config, "enable_wandb", True) and not is_wandb_sweep:
        create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)

    set_random_seed(config.seed)

    agent = build_agent(config)

    cli_init_from_run = getattr(args, "init_from_run", None)
    init_from_run = cli_init_from_run if cli_init_from_run is not None else config.init_from_run
    if init_from_run:
        _load_pretrained_weights(agent, init_from_run)

    agent.learn()
    print_training_completion(agent)
