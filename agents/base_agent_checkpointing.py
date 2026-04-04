"""BaseAgent helpers for checkpoint persistence and restoration."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def restore_deferred_optimizer_states(agent: "BaseAgent") -> None:
    """Restore optimizer state that had to wait for trainer initialization."""
    if not hasattr(agent, "_deferred_optimizer_states"):
        return

    optimizer_states = agent._deferred_optimizer_states
    optimizers = agent.optimizers()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    for opt, opt_state in zip(optimizers, optimizer_states):
        opt.load_state_dict(opt_state)
    print("Optimizer state restored from checkpoint")
    delattr(agent, "_deferred_optimizer_states")


def save_agent_checkpoint(agent: "BaseAgent", checkpoint_dir: Path) -> None:
    """Save model, optimizer, counters, RNG state, and config to a checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)

    model_path = checkpoint_dir / "model.pt"
    torch.save(agent.policy_model.state_dict(), model_path)

    try:
        optimizers = agent.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        optimizer_states = [opt.state_dict() for opt in optimizers]
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer_states, optimizer_path)
    except RuntimeError:
        pass

    train_collector = agent.get_rollout_collector("train")
    train_metrics = train_collector.get_metrics()
    val_collector = agent.get_rollout_collector("val")

    config_dict = asdict(agent.config)
    config_dict["algo_id"] = agent.config.algo_id

    state = {
        "epoch": int(agent.current_epoch),
        "total_env_steps": train_metrics.get("cnt/total_env_steps", 0),
        "total_vec_steps": train_metrics.get("cnt/total_vec_steps", 0),
        "run_id": agent.run.run_id if agent.run else None,
        "config": config_dict,
        "best_train_reward": float(train_collector._best_episode_reward),
        "best_val_reward": float(val_collector._best_episode_reward),
        "rng_states": {
            "torch": torch.get_rng_state().tolist(),
            "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None,
            "numpy": {
                "state_type": np.random.get_state()[0],
                "state_keys": np.random.get_state()[1].tolist(),
                "state_pos": int(np.random.get_state()[2]),
                "state_has_gauss": int(np.random.get_state()[3]),
                "state_cached_gaussian": float(np.random.get_state()[4]),
            },
            "random": random.getstate(),
        },
    }

    from utils.io import write_json

    state_path = checkpoint_dir / "state.json"
    write_json(state_path, state)


def load_agent_checkpoint(
    agent: "BaseAgent",
    checkpoint_dir: Path,
    *,
    resume_training: bool = True,
    strict: bool = True,
    load_optimizer_only: bool = False,
) -> None:
    """Restore model weights and optional training state from a checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "model.pt"
    old_model_path = checkpoint_dir / "policy.ckpt"

    if model_path.exists():
        model_state = torch.load(model_path, map_location="cpu", weights_only=True)
        _load_state_dict_flexible(agent, model_state, strict)
    elif old_model_path.exists():
        print("Loading from old checkpoint format (policy.ckpt)")
        checkpoint = torch.load(old_model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            _load_state_dict_flexible(agent, checkpoint["model_state_dict"], strict)
        else:
            _load_state_dict_flexible(agent, checkpoint, strict)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path} or {old_model_path}",
        )

    state_path = checkpoint_dir / "state.json"
    if state_path.exists():
        from utils.io import read_json

        state = read_json(state_path)
    else:
        print("Warning: Checkpoint missing state.json, skipping optimizer/RNG restoration")
        state = None
        resume_training = False

    if resume_training and state:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            try:
                optimizer_states = torch.load(
                    optimizer_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(optimizer_states, list):
                    optimizer_states = [optimizer_states]

                optimizers = agent.optimizers()
                if not isinstance(optimizers, (list, tuple)):
                    optimizers = [optimizers]

                for opt, opt_state in zip(optimizers, optimizer_states):
                    opt.load_state_dict(opt_state)
                print("Optimizer state restored")
            except RuntimeError:
                agent._deferred_optimizer_states = optimizer_states
                print("Note: Optimizer state will be restored after trainer initialization")

        if not load_optimizer_only:
            _restore_rng_states(state["rng_states"])

            train_collector = agent.get_rollout_collector("train")
            if state.get("best_train_reward") is not None:
                train_collector._best_episode_reward = float(state["best_train_reward"])
            if "total_env_steps" in state:
                train_collector.total_steps = int(state["total_env_steps"])
            elif "total_timesteps" in state:
                train_collector.total_steps = int(state["total_timesteps"])
            if "total_vec_steps" in state:
                train_collector.total_vec_steps = int(state["total_vec_steps"])

            if state.get("best_val_reward") is not None:
                val_collector = agent.get_rollout_collector("val")
                val_collector._best_episode_reward = float(state["best_val_reward"])

    if state:
        epoch = state.get("epoch", "unknown")
        total_env_steps = state.get(
            "total_env_steps",
            state.get("total_timesteps", "unknown"),
        )
        best_train = state.get("best_train_reward", "unknown")
        best_val = state.get("best_val_reward", "unknown")

        print(f"Checkpoint loaded from epoch {epoch}:")
        print(f"  Total env steps: {total_env_steps}")
        print(f"  Best train reward: {best_train}")
        print(f"  Best val reward: {best_val}")
    else:
        print("Checkpoint loaded (model weights only, no training state)")


def _load_state_dict_flexible(agent: "BaseAgent", state_dict, strict: bool):
    """Load state dict with optional filtering for transfer-learning scenarios."""
    if strict:
        return agent.policy_model.load_state_dict(state_dict, strict=True)

    model_state = agent.policy_model.state_dict()
    filtered_state = {}
    size_mismatches = []

    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                size_mismatches.append(key)

    result = agent.policy_model.load_state_dict(filtered_state, strict=False)
    loaded = len(filtered_state)
    skipped = len(size_mismatches)
    missing = len(result.missing_keys)

    if loaded > 0:
        print(
            "Partial weight loading: "
            f"{loaded} params loaded, {skipped} skipped (size mismatch), {missing} missing",
        )
    else:
        print("Warning: No compatible weights found for transfer learning")

    return result


def _restore_rng_states(rng_states: dict) -> None:
    """Restore torch, NumPy, and Python RNG states from serialized checkpoint data."""
    torch.set_rng_state(torch.ByteTensor(rng_states["torch"]))
    if rng_states.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(s) for s in rng_states["torch_cuda"]],
        )

    numpy_state = rng_states["numpy"]
    np_state_tuple = (
        numpy_state["state_type"],
        np.array(numpy_state["state_keys"], dtype=np.uint32),
        numpy_state["state_pos"],
        numpy_state["state_has_gauss"],
        numpy_state["state_cached_gaussian"],
    )
    np.random.set_state(np_state_tuple)

    random_state = rng_states["random"]
    random.setstate((random_state[0], tuple(random_state[1]), random_state[2]))
