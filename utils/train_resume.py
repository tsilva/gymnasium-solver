"""Resume and pretrained-weight helpers for training flows."""

from __future__ import annotations

from dataclasses import asdict

import wandb

from agents import build_agent

from utils.config import Config
from utils.random import set_random_seed
from utils.run import load_available_run, resolve_checkpoint_dir, resolve_run_id
from utils.train_overrides import apply_cli_overrides
from utils.train_reporting import print_training_completion


def launch_training_resume(args) -> None:
    """Launch training in resume mode from an existing checkpoint."""
    from utils.io import read_json

    run_id = args.resume
    print(f"Resuming run: {run_id}")
    run = load_available_run(run_id)
    run_id = run.run_id

    checkpoint_dir = resolve_checkpoint_dir(run, args.epoch)
    print(f"Loading checkpoint from: {checkpoint_dir}")

    state_path = checkpoint_dir / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        config_dict = state.get("config")
        if config_dict and "algo_id" in config_dict:
            config = Config.build_from_dict(config_dict)
        else:
            print("Warning: Checkpoint uses old format, loading config from run directory")
            config = run.load_config()
            state = None
    else:
        print("Warning: Checkpoint missing state.json, loading config from run directory")
        config = run.load_config()
        state = None

    config = apply_cli_overrides(config, args)

    if getattr(config, "enable_wandb", True):
        project_name = config.project_id
        assert project_name, "project_id is required"
        wandb.init(
            project=project_name,
            id=run_id,
            name=run_id,
            resume="must",
            config=asdict(config),
        )

    set_random_seed(config.seed)

    agent = build_agent(config)
    agent.run = run
    agent.load_checkpoint(checkpoint_dir, resume_training=True)

    if state:
        loaded_epoch = state.get("epoch", 0)
        agent._resume_from_epoch = loaded_epoch
        print(f"Continuing training from epoch {loaded_epoch}")
    else:
        print("Warning: Cannot determine checkpoint epoch, starting from 0")

    agent.learn()
    print_training_completion(agent)


def load_pretrained_weights(agent, run_spec: str, load_optimizer: bool = True) -> None:
    """Load pretrained weights from another run's checkpoint."""
    if "/" in run_spec:
        run_id, checkpoint_spec = run_spec.split("/", 1)
    else:
        run_id = run_spec
        checkpoint_spec = None

    resolved_run_id = resolve_run_id(run_id)
    print(f"Loading pretrained weights from run: {resolved_run_id}")
    run = load_available_run(run_id)
    checkpoint_dir = resolve_checkpoint_dir(run, checkpoint_spec)
    checkpoint_desc = checkpoint_spec if checkpoint_spec else "(@best if available, else @last)"
    print(f"Loading weights from: {checkpoint_dir} {checkpoint_desc}")

    agent.load_checkpoint(
        checkpoint_dir,
        resume_training=load_optimizer,
        strict=False,
        load_optimizer_only=load_optimizer,
    )
    if load_optimizer:
        print(f"Pretrained weights and optimizer state loaded from {resolved_run_id}")
        print("Note: Starting fresh training progress (epoch 0, timestep 0) with warm optimizer")
    else:
        print(f"Pretrained weights loaded from {resolved_run_id} (optimizer state not loaded)")
