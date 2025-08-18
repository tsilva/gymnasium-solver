import argparse
from typing import Optional

from utils.config import load_config

try:
    import torch  # For device availability checks when accelerator is 'auto'
except Exception:  # pragma: no cover - training script fallback
    torch = None  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID (default: Pong-v5_ram_ppo)")
    parser.add_argument("--algo", type=str, default=None, help="Agent type (optional, extracted from config if not provided)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run non-interactively: auto-accept prompts and defaults")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
    # Override resume flag if provided via command line
    if args.resume:
        config.resume = True
    # Quiet mode disables interactive prompts
    if getattr(args, "quiet", False):
        setattr(config, "quiet", True)
    
    # Group session header and config/agent details neatly once
    print("=== Training Session Started ===")

    # Warn early if using CNN policy on CPU (can be very slow)
    def _is_cpu_training(cfg) -> bool:
        acc: Optional[str] = getattr(cfg, "accelerator", "cpu")
        if acc == "cpu":
            return True
        if acc in ("auto", None):
            # If no GPU/MPS is available, this will end up on CPU
            try:
                has_cuda = bool(torch and torch.cuda.is_available())
                has_mps = bool(torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
                return not (has_cuda or has_mps)
            except Exception:
                return True
        return False

    pol = getattr(config, "policy", "")
    if isinstance(pol, str) and pol.lower() == "cnnpolicy" and _is_cpu_training(config):
        print("[WARN] CnnPolicy selected but training device is CPU. CNN training on CPU can be very slow. "
              "Consider setting accelerator='gpu' (or 'auto') and devices='auto' if a GPU is available.")
    
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    from agents import create_agent
    agent = create_agent(config)

    # Print config details once here (will also be captured to logs)
    from utils.logging import log_config_details
    log_config_details(config)

    # Print model details once
    print(str(agent))
    agent.learn()
    print("Training completed.")
        
if __name__ == "__main__":
    main()
