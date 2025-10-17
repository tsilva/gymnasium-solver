import argparse
import sys
import warnings

import torch

from utils.train_launcher import launch_training_from_args
from utils.environment_registry import list_available_environments

# Suppress PyTorch Lightning's num_workers warning.
# num_workers=0 is intentional: we keep rollout tensors in memory and
# DataLoader only indexes them, so multi-process workers add IPC overhead
# without benefit.
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# Suppress Pydantic warnings when Lightning serializes Config for Modal.
# These occur when Pydantic auto-converts dataclass fields with Union types and repr=False.
# The warnings are harmless but noisy - functionality is not affected.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

# Suppress PyTorch TF32 API deprecation warnings.
# These are informational notices about old API being deprecated after PyTorch 2.9.
warnings.filterwarnings("ignore", message=".*TF32.*")

# Set matmul precision for Tensor Cores on CUDA devices (RTX 4090, etc.)
# 'medium' balances precision and performance on GPUs with Tensor Cores
torch.set_float32_matmul_precision("medium")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument(
        "config", 
        nargs="?",
        help="Config spec '<env>:<variant>' (e.g., CartPole-v1:ppo) or environment name for fuzzy search"
    )
    parser.add_argument(
        "--config_id", 
        type=str, 
        default=None, 
        help="Config ID '<env>:<variant>' (e.g., CartPole-v1:ppo)"
    )
    parser.add_argument(
        "--wandb_sweep", 
        action="store_true", 
        default=False, 
        help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training."
    )
    parser.add_argument(
        "--max-env-steps",
        dest="max_env_steps",
        default=None,
        help="Override config max_env_steps (total environment steps/frames). Accepts integers or scientific notation.",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override any config field (e.g., --override policy_lr=0.001 --override batch_size=64). Can be specified multiple times.",
    )
    parser.add_argument(
        "--env-kwargs",
        action="append",
        dest="env_kwargs",
        metavar="KEY=VALUE",
        help="Override env_kwargs fields (e.g., --env-kwargs state=Level2-1). Can be specified multiple times.",
    )
    parser.add_argument(
        "--list-envs",
        nargs="?",
        const="",
        metavar="SEARCH",
        help="List all available environment targets with descriptions and exit. Optionally provide a search term to filter environments (e.g., 'Pong' or 'CartPole')."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume training from a checkpoint. Use run ID (e.g., 'abc123') or '@last' for most recent run."
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default=None,
        metavar="EPOCH",
        help="Specific epoch to resume from. Use epoch number, '@best', or '@last' (default: '@best' if exists, else '@last')."
    )
    parser.add_argument(
        "--init-from-run",
        type=str,
        default=None,
        metavar="RUN_SPEC",
        help="Initialize model weights from another run's checkpoint. Format: 'run_id' or 'run_id/checkpoint'. Checkpoint can be '@best', '@last', or 'epoch=N'. If not specified, uses '@best' if available, else '@last'. Downloads from W&B if not found locally. Examples: 'abc123', 'abc123/@best', 'abc123/epoch=13', '@last/@best'."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "modal"],
        default="local",
        help="Training backend to use. 'local' runs training on current machine (default), 'modal' runs remotely on Modal AI infrastructure (requires Modal account and token configured)."
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        default=False,
        help="When using --backend modal, run training in detached mode (job continues if terminal is closed). Default is to stream logs and block until completion."
    )
    args = parser.parse_args()

    # If backend is modal, launch training on Modal instead of locally
    if args.backend == "modal":
        try:
            from utils.modal_train import launch_modal_training
            launch_modal_training(args)
            return
        except ImportError:
            print("Error: Modal dependencies not installed. Install with: uv pip install -e '.[modal]'")
            sys.exit(1)

    # In case list envs flag is passed, use argument as search
    # term and show all available environments matching it
    # (exits after printing)
    if args.list_envs is not None:
        search_term = args.list_envs if args.list_envs else None
        list_available_environments(search_term)
        return

    # If resuming, skip environment search
    if args.resume:
        launch_training_from_args(args)
        return

    # Otherwise, check if we need to search for environment
    config_id = args.config or args.config_id or "Bandit-v0:ppo" # TODO: move default up
    should_search = ":" not in config_id
    if should_search:
        list_available_environments(config_id)
        return

    # Parse args and start training
    launch_training_from_args(args)
        
if __name__ == "__main__":
    main()
