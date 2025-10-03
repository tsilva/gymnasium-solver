"""Keyboard shortcut support during training with pause/resume."""

import sys
import select
import threading
import tty
import termios
from typing import Optional

import pytorch_lightning as pl


class KeyboardShortcutCallback(pl.Callback):
    """Listen for keyboard shortcuts during training and execute actions.

    Pauses training when a key is pressed to handle interactive prompts,
    then resumes after action completes.

    Supported shortcuts:
    - [c]: Force checkpoint - runs validation and saves checkpoint regardless of eval schedule
    """

    def __init__(self):
        super().__init__()
        self._input_thread: Optional[threading.Thread] = None
        self._stop_thread = threading.Event()
        self._pending_action: Optional[str] = None
        self._action_lock = threading.Lock()
        self._trainer: Optional[pl.Trainer] = None
        self._agent: Optional[pl.LightningModule] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Start keyboard listener thread when training begins."""
        self._trainer = trainer
        self._agent = pl_module
        self._stop_thread.clear()

        # Only start listener if stdin is a TTY (not in pipes/redirects)
        if sys.stdin.isatty():
            self._input_thread = threading.Thread(
                target=self._listen_for_input,
                daemon=True,
                name="KeyboardShortcutListener"
            )
            self._input_thread.start()
            print("\n[Keyboard shortcuts enabled: press 'c' to force checkpoint, 'h' for help]\n")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Stop keyboard listener thread when training ends."""
        self._stop_thread.set()
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int
    ):
        """Check for pending actions before each batch."""
        with self._action_lock:
            if self._pending_action is not None:
                action = self._pending_action
                self._pending_action = None

                # Pause training to handle action
                print(f"\n[Training paused - handling shortcut '{action}']")
                self._handle_action(action, trainer, pl_module)
                print("[Training resumed]\n")

    def _listen_for_input(self):
        """Background thread that listens for keyboard input."""
        if sys.platform != 'win32':
            # Unix-like systems: temporarily set raw mode only when reading
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                while not self._stop_thread.is_set():
                    # Use select to check if input is available (with timeout)
                    ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if ready:
                        # Set raw mode only for reading
                        tty.setraw(fd)
                        char = sys.stdin.read(1)
                        # Immediately restore normal mode
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        if char:
                            with self._action_lock:
                                self._pending_action = char.lower()
            finally:
                # Always restore terminal settings
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else:
            # Windows - already works without Enter using msvcrt.getch()
            import msvcrt
            while not self._stop_thread.is_set():
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8', errors='ignore')
                    with self._action_lock:
                        self._pending_action = char.lower()
                else:
                    self._stop_thread.wait(timeout=0.5)

    def _handle_action(self, action: str, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Handle a keyboard shortcut action."""

        if action == 'c':
            self._force_checkpoint(trainer, pl_module)
        elif action == 'h':
            self._show_help()
        else:
            print(f"[Unknown shortcut '{action}' - press 'h' for help]")

    def _force_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Force a validation run and checkpoint save."""
        print("\n=== Forced Checkpoint ===")
        print("Running validation...")

        # Get validation environment and collector
        val_env = pl_module.get_env("val")
        val_collector = pl_module.get_rollout_collector("val")

        # Run evaluation episodes
        with val_env.recorder(video_path="", record_video=False):
            val_results = val_collector.evaluate_episodes(
                n_episodes=pl_module.config.eval_episodes,
                deterministic=pl_module.config.eval_deterministic,
            )

        # Log validation metrics
        val_metrics = val_collector.get_metrics()
        for key, value in val_metrics.items():
            if key.startswith("val/"):
                trainer.logged_metrics[key] = value

        ep_rew_mean = val_metrics.get('val/roll/ep_rew/mean', None)
        if ep_rew_mean is not None:
            print(f"Validation complete: ep_rew_mean = {ep_rew_mean:.2f}")
        else:
            print("Validation complete: ep_rew_mean = N/A")

        # Find checkpoint callback and trigger save
        from trainer_callbacks.model_checkpoint import ModelCheckpointCallback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpointCallback):
                checkpoint_callback = callback
                break

        if checkpoint_callback is not None:
            print("Saving checkpoint...")
            checkpoint_callback._if_best_epoch_save_checkpoint(trainer, pl_module)
            print("Checkpoint saved!")
        else:
            print("Warning: ModelCheckpointCallback not found, cannot save checkpoint")

        print("=========================\n")

    def _show_help(self):
        """Display help message with available shortcuts."""
        print("\n=== Keyboard Shortcuts ===")
        print("  c : Force checkpoint (run validation + save)")
        print("  h : Show this help message")
        print("==========================\n")
