import argparse
import os
from dataclasses import asdict

from utils.config import load_config, Config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config_id", type=str, default="CartPole-v1:ppo", help="Config ID (e.g., CartPole-v1_ppo)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Run non-interactively: auto-accept prompts and defaults")
    parser.add_argument(
        "--wandb_sweep", action="store_true", default=False,
        help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training."
    )
    args = parser.parse_args()

    # Load configuration
    config_id = args.config_id
    #config_id = "ALE-Pong-v5_ram:ppo"
    config_id, variant_id = config_id.split(":")
    config = load_config(config_id, variant_id)

    # Apply args to config
    if args.quiet is True: config.quiet = True

    # If running under a W&B agent or explicitly requested, initialize wandb early
    # and merge sweep overrides into our Config before creating the agent.
    use_wandb_sweep = bool(args.wandb_sweep)
    # Auto-detect when launched by wandb agent
    try:
        use_wandb_sweep = use_wandb_sweep or bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"))
    except Exception:
        pass

    if use_wandb_sweep:
        try:
            import wandb  # type: ignore
            # Local import to avoid heavy deps at import-time
            try:
                from utils.dict_utils import convert_dict_numeric_strings  # type: ignore
            except Exception:
                convert_dict_numeric_strings = lambda d: d  # no-op fallback

            # Initialize W&B without overriding project/name to respect sweep settings
            defaults = asdict(config)
            wandb.init(config=defaults)

            # Merge wandb.config into our dataclass; this lets sweeps override any field
            # present in the dataclass while preserving non-overridden defaults.
            try:
                sweep_cfg = dict(wandb.config)
            except Exception:
                # Fallback for older wandb versions
                sweep_cfg = getattr(wandb.config, "as_dict", lambda: {} )()

            # Parse linear schedules like lin_0.001 into value + *_schedule
            if isinstance(sweep_cfg, dict):
                # Coerce numeric-like strings first (e.g., "3e-5")
                convert_dict_numeric_strings(sweep_cfg)
                Config._parse_schedules(sweep_cfg)

            # Start from the current config values, overlay sweep overrides
            merged = asdict(config)
            merged.update(sweep_cfg if isinstance(sweep_cfg, dict) else {})
            # Ensure merged dict also has numeric strings coerced (robustness)
            try:
                convert_dict_numeric_strings(merged)
            except Exception:
                pass

            # Resolve fractional batch_size (0 < x <= 1 → fraction of n_envs * n_steps)
            try:
                bs = merged.get("batch_size")
                if isinstance(bs, float) and bs > 0 and bs <= 1:
                    ne = int(merged.get("n_envs", config.n_envs))
                    ns = int(merged.get("n_steps", config.n_steps or 1))
                    merged["batch_size"] = max(1, int(ne * ns * float(bs)))
            except Exception:
                # Be robust: if conversion fails, keep original value
                pass

            # Instantiate the correct algo-specific Config subclass based on algo_id
            algo_id = str(merged.get("algo_id", config.algo_id)).lower()
            ConfigClass = {
                "qlearning": Config.__subclasses__()[0] if any(c.__name__ == 'QLearningConfig' for c in Config.__subclasses__()) else Config,
                "reinforce": Config.__subclasses__()[1] if any(c.__name__ == 'REINFORCEConfig' for c in Config.__subclasses__()) else Config,
                "ppo": Config.__subclasses__()[2] if any(c.__name__ == 'PPOConfig' for c in Config.__subclasses__()) else Config,
            }.get(algo_id, config.__class__)

            # Avoid relying on subclass index order; fetch by name if available
            try:
                from utils.config import PPOConfig, REINFORCEConfig, QLearningConfig  # type: ignore
                ConfigClass = {"ppo": PPOConfig, "reinforce": REINFORCEConfig, "qlearning": QLearningConfig}.get(algo_id, config.__class__)
            except Exception:
                pass

            # Rebuild the config instance and validate
            config = ConfigClass(**merged)  # type: ignore[call-arg]
            try:
                config.validate()
            except Exception:
                # If validation isn't available or fails unexpectedly, continue with merged values
                pass
        except Exception as e:
            # If wandb is missing or init fails, proceed without sweep overrides
            print(f"Warning: W&B sweep initialization failed: {e}. Continuing without sweep overrides.")

    # Normalize enum-like string aliases to canonical values expected downstream
    def _canon_returns(x):
        try:
            # Enum instance → its value
            return str(x.value)
        except Exception:
            pass
        s = str(x) if x is not None else None
        if s is None:
            return None
        s_low = s.lower().replace(" ", "")
        s_low = s_low.replace("_", ":").replace("-", ":")
        # Map common aliases
        if s_low in {"gae:rtg", "gae:returns_to_go", "gae"+":rtg"}:
            return "gae:rtg"
        if s_low in {"mc:rtg", "rtg", "reward_to_go"}:
            return "mc:rtg"
        if s_low in {"mc:episode", "episode", "full_episode"}:
            return "mc:episode"
        return s

    def _canon_adv(x):
        try:
            return str(x.value)
        except Exception:
            pass
        s = str(x) if x is not None else None
        if s is None:
            return None
        s_low = s.lower()
        if s_low in {"gae"}:
            return "gae"
        if s_low in {"baseline", "value_baseline"}:
            return "baseline"
        return s

    try:
        rt = getattr(config, "returns_type", None)
        at = getattr(config, "advantages_type", None)
        if rt is not None:
            setattr(config, "returns_type", _canon_returns(rt))
        if at is not None:
            setattr(config, "advantages_type", _canon_adv(at))
    except Exception:
        pass

    # TODO: move this out of here
    # Set global random seed
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    # Create agent and start learning
    from agents import create_agent
    agent = create_agent(config)
    agent.learn()
        
if __name__ == "__main__":
    main()
