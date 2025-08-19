"""Run directory management utilities for organizing all run assets."""

from pathlib import Path

class RunManager:
    """Manages run-specific directories and assets organization."""

    def __init__(self, run_id :str, base_runs_dir: str = "runs"):
        # Ensure run dir exists
        self.base_runs_dir = Path(base_runs_dir)
        self.run_id = run_id
        self.run_dir = self.base_runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Make `latest-run` symlink to this run
        latest = self.base_runs_dir / "latest-run"
        if latest.exists() or latest.is_symlink(): latest.unlink()
        latest.symlink_to(Path(self.run_id))
    
    def ensure_dir(self, subdir: str) -> Path:
        """Ensure a subdirectory exists within the run directory."""
        dir_path = self.run_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def print_details(self):
         # Ask for confirmation before any heavy setup (keep prior prints grouped)
        # Before prompting, suggest better defaults if we detect mismatches
        self._maybe_warn_observation_policy_mismatch()

        self._print_env_spec(self.train_env)

        print("Starting training...")
        print(f"Run directory: {self.run_dir}")
        print(f"Run ID: {self.run_manager.run_id}")

    # TODO: move this somewhere else?
    def _maybe_warn_observation_policy_mismatch(self):
        from utils.environment import is_rgb_env
        
        # In case the observation space is RGB, warn if MLP policy is used
        is_rgb = is_rgb_env(self.train_env)
        is_mlp = self.config.policy.lower() == "mlp"
        if is_rgb and is_mlp:
            print(
                "Warning: Detected RGB image observations with MLP policy. "
                "For pixel inputs, consider using CNN for better performance."
            )

        # In case the observation space is not RGB, warn if CNN policy is used
        is_cnn = self.config.policy.lower() == "cnn"
        if not is_rgb and is_cnn:
            print(
                "Warning: Detected non-RGB observations with CNN policy. "
                "For non-image inputs, consider using MLP for better performance."
            )
    
    # TODO: print_env() direclty in env
    def _print_env_spec(self, env):
        # Show environment details for transparency
        print("\n=== Environment Details ===")
        
        # Observation space and action space from vectorized env
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Reward range and threshold when available
        reward_range = env.get_reward_range()
        print(f"Reward range: {reward_range}")

        # Reward threshold if defined
        reward_threshold = env.get_reward_threshold()
        print(f"Reward threshold: {reward_threshold}")
        print("=" * 30)
