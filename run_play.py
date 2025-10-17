#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform
import time

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.random import set_random_seed
from utils.rollouts import RolloutCollector
from utils.run import Run


class RewardPlotter:
    """Real-time reward plotter for visualizing rewards during playback."""

    def __init__(self, update_interval: float = 0.2):
        """Initialize the plotter with two subplots for episode and step rewards.

        Args:
            update_interval: Minimum time (in seconds) between plot updates to throttle rendering
        """
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore
            import time

            self.pg = pg
            self.time = time

            # Set background to white for better visibility
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

            # Create window with two plots
            self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Reward Visualization")
            self.win.resize(1000, 800)
            self.win.setWindowTitle('Real-time Reward Visualization')

            # Position window on the right side of the screen to not overlap with game
            # Game window is typically on the left, so position plot on the right
            self.win.move(1050, 50)  # x=1050 (right side), y=50 (near top)

            # Episode reward plot (accumulated reward over steps)
            self.plot_episode = self.win.addPlot(title="Accumulated Reward per Episode")
            self.plot_episode.setLabel('left', 'Accumulated Episode Reward')
            self.plot_episode.setLabel('bottom', 'Step')
            self.plot_episode.showGrid(x=True, y=True, alpha=0.3)
            self.plot_episode.addLegend()

            # Move to next row
            self.win.nextRow()

            # Step reward plot (individual step rewards)
            self.plot_step = self.win.addPlot(title="Individual Step Rewards")
            self.plot_step.setLabel('left', 'Step Reward')
            self.plot_step.setLabel('bottom', 'Step')
            self.plot_step.showGrid(x=True, y=True, alpha=0.3)

            # Add zero line
            self.plot_step.addLine(y=0, pen=pg.mkPen('k', width=1, style=QtCore.Qt.PenStyle.DashLine))

            # Data storage
            self.episode_data = []  # List of (steps, rewards, curve) for each episode
            self.current_episode_steps = []
            self.current_episode_rewards = []
            self.current_step_rewards = []
            self.global_step = 0
            self.episode_num = 0
            self.current_curve = None

            # Step reward bars
            self.step_bars = None

            # Color palette for different episodes (RGB tuples)
            self.colors = [
                (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
            ]

            # Track if window is still open
            self.is_open = True

            # Throttling: only update plot at fixed intervals
            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            self.use_pyqtgraph = True

        except ImportError:
            # Fallback to matplotlib if pyqtgraph is not available
            import matplotlib.pyplot as plt
            import time

            self.plt = plt
            self.time = time
            self.plt.ion()

            self.fig, (self.ax_episode, self.ax_step) = self.plt.subplots(2, 1, figsize=(10, 8))
            self.fig.suptitle('Real-time Reward Visualization', fontsize=14, fontweight='bold')

            self.ax_episode.set_xlabel('Step')
            self.ax_episode.set_ylabel('Accumulated Episode Reward')
            self.ax_episode.set_title('Accumulated Reward per Episode')
            self.ax_episode.grid(True, alpha=0.3)

            self.ax_step.set_xlabel('Step')
            self.ax_step.set_ylabel('Step Reward')
            self.ax_step.set_title('Individual Step Rewards')
            self.ax_step.grid(True, alpha=0.3)
            self.ax_step.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

            self.episode_data = []
            self.current_episode_steps = []
            self.current_episode_rewards = []
            self.current_step_rewards = []
            self.global_step = 0
            self.episode_num = 0

            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            self.is_open = True
            self.fig.canvas.mpl_connect('close_event', self._on_close)

            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            self.plt.tight_layout()
            self.plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.use_pyqtgraph = False

    def _on_close(self, event):
        """Handle window close event."""
        self.is_open = False

    def add_step(self, step_reward: float):
        """Add a step reward and update the plots (throttled).

        Args:
            step_reward: Reward received at this step
        """
        if not self.is_open:
            return

        # Update current episode data
        self.current_episode_steps.append(self.global_step)
        accumulated_reward = (self.current_episode_rewards[-1] if self.current_episode_rewards else 0) + step_reward
        self.current_episode_rewards.append(accumulated_reward)
        self.current_step_rewards.append(step_reward)
        self.global_step += 1

        # Mark that we need an update
        self.needs_update = True

        # Only update plots if enough time has passed (throttling)
        current_time = self.time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._update_plots()
            self.last_update_time = current_time
            self.needs_update = False

            # Process events for pyqtgraph
            if self.use_pyqtgraph:
                from pyqtgraph.Qt import QtWidgets
                QtWidgets.QApplication.processEvents()

    def reset_episode(self):
        """Reset for a new episode and force a plot update."""
        if not self.is_open:
            return

        # Store completed episode data
        if self.current_episode_steps:
            if self.use_pyqtgraph:
                # Store the curve reference for pyqtgraph
                self.episode_data.append((
                    self.current_episode_steps.copy(),
                    self.current_episode_rewards.copy(),
                    self.current_curve
                ))
            else:
                self.episode_data.append((self.current_episode_steps.copy(), self.current_episode_rewards.copy()))

        # Reset for next episode
        self.current_episode_steps = []
        self.current_episode_rewards = []
        self.current_curve = None
        self.episode_num += 1

        # Force an update at episode boundaries
        if self.needs_update:
            self._update_plots()
            self.last_update_time = self.time.time()
            self.needs_update = False

            # Process events for pyqtgraph
            if self.use_pyqtgraph:
                from pyqtgraph.Qt import QtWidgets
                QtWidgets.QApplication.processEvents()

    def _update_plots(self):
        """Update both subplots with current data."""
        if not self.is_open:
            return

        try:
            if self.use_pyqtgraph:
                # PyQtGraph (fast) implementation
                import numpy as np

                # Update current episode curve
                if self.current_episode_steps:
                    color = self.colors[self.episode_num % len(self.colors)]
                    if self.current_curve is None:
                        # Create new curve for this episode
                        self.current_curve = self.plot_episode.plot(
                            self.current_episode_steps,
                            self.current_episode_rewards,
                            pen=self.pg.mkPen(color=color, width=2),
                            name=f'Episode {self.episode_num+1}'
                        )
                    else:
                        # Update existing curve (much faster than recreating)
                        self.current_curve.setData(self.current_episode_steps, self.current_episode_rewards)

                # Update step rewards (use bar graph for better performance)
                if self.current_step_rewards:
                    x = np.array(self.current_episode_steps)
                    y = np.array(self.current_step_rewards)

                    # Clear previous bars if needed
                    if self.step_bars is not None:
                        self.plot_step.removeItem(self.step_bars)

                    # Create bar graph item
                    self.step_bars = self.pg.BarGraphItem(
                        x=x, height=y, width=0.8,
                        brush=self.pg.mkBrush(70, 130, 180, 150)
                    )
                    self.plot_step.addItem(self.step_bars)

            else:
                # Matplotlib (fallback) implementation
                # Clear plots
                self.ax_episode.clear()
                self.ax_step.clear()

                # Reapply labels and formatting
                self.ax_episode.set_xlabel('Step')
                self.ax_episode.set_ylabel('Accumulated Episode Reward')
                self.ax_episode.set_title('Accumulated Reward per Episode')
                self.ax_episode.grid(True, alpha=0.3)

                self.ax_step.set_xlabel('Step')
                self.ax_step.set_ylabel('Step Reward')
                self.ax_step.set_title('Individual Step Rewards')
                self.ax_step.grid(True, alpha=0.3)
                self.ax_step.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

                # Plot completed episodes
                for i, (steps, rewards) in enumerate(self.episode_data):
                    color = self.colors[i % len(self.colors)]
                    self.ax_episode.plot(steps, rewards, color=color, linewidth=2, alpha=0.7, label=f'Episode {i+1}')

                # Plot current episode
                if self.current_episode_steps:
                    color = self.colors[self.episode_num % len(self.colors)]
                    self.ax_episode.plot(self.current_episode_steps, self.current_episode_rewards,
                                        color=color, linewidth=2, label=f'Episode {self.episode_num+1} (current)')

                # Plot step rewards
                all_steps = []
                all_step_rewards = []
                episode_boundaries = []

                # Collect all step data
                for episode_steps, _ in self.episode_data:
                    if episode_steps:
                        episode_boundaries.append(episode_steps[-1])

                # Add current episode step rewards
                if self.current_step_rewards:
                    all_steps.extend(self.current_episode_steps)
                    all_step_rewards.extend(self.current_step_rewards)

                if all_steps:
                    self.ax_step.bar(all_steps, all_step_rewards, width=1.0, alpha=0.6, color='steelblue')

                # Add episode boundary lines
                for boundary in episode_boundaries:
                    self.ax_episode.axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    self.ax_step.axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)

                # Add legends
                if len(self.episode_data) + (1 if self.current_episode_steps else 0) > 0:
                    self.ax_episode.legend(loc='best', fontsize=8)

                self.plt.tight_layout()
                # Draw the canvas and flush events without blocking
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        except Exception:
            # Ignore plot update errors (e.g., if window was closed)
            self.is_open = False

    def close(self):
        """Close the plot window, flushing any pending updates first."""
        if self.is_open:
            # Flush any pending updates before closing
            if self.needs_update:
                try:
                    self._update_plots()
                except Exception:
                    pass

            try:
                if self.use_pyqtgraph:
                    self.win.close()
                else:
                    self.plt.close(self.fig)
            except Exception:
                pass
            self.is_open = False


def extract_action_labels_from_config(config) -> dict[int, str] | None:
    """Extract and remap action labels from config spec.

    Returns a dict mapping action_id (after wrappers) to label string.
    Returns None if labels are not available in config.
    """
    # Check if config has spec with action_space labels
    if not hasattr(config, 'spec') or config.spec is None:
        return None

    spec = config.spec
    if not isinstance(spec, dict) or 'action_space' not in spec:
        return None

    action_space_spec = spec['action_space']
    if not isinstance(action_space_spec, dict) or 'labels' not in action_space_spec:
        return None

    original_labels = action_space_spec['labels']
    if not isinstance(original_labels, dict):
        return None

    # Check if there's a DiscreteActionSpaceRemapperWrapper
    remapping = None
    if hasattr(config, 'env_wrappers') and config.env_wrappers:
        for wrapper_spec in config.env_wrappers:
            if isinstance(wrapper_spec, dict) and wrapper_spec.get('id') == 'DiscreteActionSpaceRemapperWrapper':
                remapping = wrapper_spec.get('mapping')
                break

    # Apply remapping if present
    if remapping:
        # remapping[i] = original_action_id
        # So new_action_id i maps to original_labels[remapping[i]]
        remapped_labels = {}
        for new_id, orig_id in enumerate(remapping):
            if orig_id in original_labels:
                remapped_labels[new_id] = original_labels[orig_id]
        return remapped_labels
    else:
        # No remapping, return original labels
        return original_labels


def play_episodes_manual(env, target_episodes: int, mode: str, step_by_step: bool = False, fps: int | None = None, action_labels: dict[int, str] | None = None, plotter: RewardPlotter | None = None):
    """Play episodes with random actions or user input."""
    import numpy as np
    import gymnasium as gym

    # Calculate frame delay for FPS limiting
    frame_delay = 1.0 / fps if fps else 0

    # Get action space info
    action_space = env.single_action_space
    is_multibinary = isinstance(action_space, gym.spaces.MultiBinary)

    if is_multibinary:
        n_actions = action_space.n  # Number of buttons
    else:
        n_actions = action_space.n  # Number of discrete actions

    # Try to import pygame for non-blocking keyboard input in GUI mode
    pygame_available = False
    pygame_screen = None
    # Track key state manually using events to avoid stuck keys
    pressed_keys = set()  # Track which keys are currently pressed

    if mode == "user" and not step_by_step:
        try:
            import pygame
            pygame.init()
            # Create a small control window to capture keyboard events
            pygame_screen = pygame.display.set_mode((400, 100))
            pygame.display.set_caption("Keyboard Controls - Press number keys (0-9)")
            pygame_available = True

            # Draw instructions on the control window
            font = pygame.font.Font(None, 24)
            pygame_screen.fill((40, 40, 40))
            text1 = font.render(f"Press keys 0-{n_actions-1} to select action", True, (255, 255, 255))
            text2 = font.render("Press Q to quit", True, (255, 255, 255))
            pygame_screen.blit(text1, (10, 25))
            pygame_screen.blit(text2, (10, 55))

            # Show action labels if available
            if action_labels:
                small_font = pygame.font.Font(None, 16)
                y_offset = 85
                for action_id in range(min(n_actions, 8)):  # Limit to first 8 actions
                    if action_id in action_labels:
                        label_text = small_font.render(f"{action_id}:{action_labels[action_id]}", True, (180, 180, 180))
                        pygame_screen.blit(label_text, (10 + (action_id % 4) * 95, y_offset + (action_id // 4) * 15))

            pygame.display.flip()
        except ImportError:
            pass

    if mode == "user":
        if step_by_step:
            if is_multibinary:
                print(f"\nAction controls: Press 0-{n_actions-1} (space-separated) for buttons, Enter to execute.")
            else:
                print(f"\nAction controls: Press 0-{n_actions-1} to select action, Enter to execute.")
            print("Step-by-step mode enabled. Press Enter to step, 'q' then Enter to quit.")
        elif pygame_available:
            print(f"\nKeyboard control window opened!")
            if is_multibinary:
                print(f"Action controls: Hold keys 0-{n_actions-1} simultaneously for multiple buttons.")
            else:
                print(f"Action controls: Press 0-{n_actions-1} keys in the control window to select action.")
            print("Press Q to quit.")
        else:
            if is_multibinary:
                print(f"\nAction controls: Press 0-{n_actions-1} (space-separated) for buttons, Enter to execute.")
            else:
                print(f"\nAction controls: Press 0-{n_actions-1} to select action, Enter to execute.")

        # Display action labels if available
        if action_labels:
            print(f"\n{'Button' if is_multibinary else 'Action'} mapping (after any wrappers):")
            for action_id in range(n_actions):
                if action_id in action_labels:
                    print(f"  {action_id}: {action_labels[action_id]}")
    else:
        if is_multibinary:
            print(f"\nRandom action mode enabled. Sampling from {n_actions} buttons (MultiBinary).")
        else:
            print(f"\nRandom action mode enabled. Sampling from {n_actions} actions.")
        if step_by_step:
            print("Step-by-step mode enabled. Press Enter to step, 'q' then Enter to quit.")

    reported_episodes = 0
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0

    # Initialize default action based on action space type
    if is_multibinary:
        action = np.zeros(n_actions, dtype=np.int8)  # All buttons off
    else:
        action = 0  # Default discrete action

    while reported_episodes < target_episodes:
        # Choose action based on mode
        if mode == "random":
            action = action_space.sample()
            if step_by_step:
                print(f"Random action: {action}")
        else:  # user mode
            if step_by_step:
                try:
                    user_input = input(f"[step {episode_length}] Action (0-{n_actions-1}): ")
                except EOFError:
                    user_input = ""

                if user_input.strip().lower() in {"q", "quit", "exit"}:
                    break

                try:
                    action = int(user_input.strip())
                    assert 0 <= action < n_actions, f"Action must be 0-{n_actions-1}"
                except (ValueError, AssertionError) as e:
                    print(f"Invalid action: {e}. Using action 0.")
                    action = 0
            elif pygame_available:
                # Process events and manually track key state to avoid stuck keys
                # This is more reliable than pygame.key.get_pressed() which can hold stale state
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if pygame_screen:
                            pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            if pygame_screen:
                                pygame.quit()
                            return
                        # Track key presses manually
                        pressed_keys.add(event.key)
                    elif event.type == pygame.KEYUP:
                        # Remove key from pressed set when released
                        pressed_keys.discard(event.key)
                    # Clear all keys on focus loss to prevent stuck keys
                    elif hasattr(pygame, 'WINDOWFOCUSLOST') and event.type == pygame.WINDOWFOCUSLOST:
                        pressed_keys.clear()
                    elif event.type == pygame.ACTIVEEVENT:
                        if hasattr(event, 'state') and event.state == 1 and hasattr(event, 'gain') and not event.gain:
                            pressed_keys.clear()

                if is_multibinary:
                    # MultiBinary: check all buttons simultaneously
                    action = np.zeros(n_actions, dtype=np.int8)
                    for i in range(min(10, n_actions)):
                        # Check main keyboard keys
                        if (pygame.K_0 + i) in pressed_keys:
                            action[i] = 1
                        # Check numpad keys
                        elif (pygame.K_KP0 + i) in pressed_keys:
                            action[i] = 1

                    # Update control window display to show active buttons
                    if pygame_screen:
                        font = pygame.font.Font(None, 24)
                        pygame_screen.fill((40, 40, 40))
                        text1 = font.render(f"Press keys 0-{n_actions-1} (hold multiple)", True, (255, 255, 255))
                        text2 = font.render("Press Q to quit", True, (255, 255, 255))

                        active_buttons = [str(i) for i in range(n_actions) if action[i]]
                        if active_buttons:
                            text3 = font.render(f"Active buttons: {','.join(active_buttons)}", True, (100, 255, 100))
                        else:
                            text3 = font.render("No buttons pressed", True, (180, 180, 180))

                        pygame_screen.blit(text1, (10, 15))
                        pygame_screen.blit(text2, (10, 45))
                        pygame_screen.blit(text3, (10, 75))
                        pygame.display.flip()
                else:
                    # Discrete: single action at a time
                    action_detected = None

                    # Check number keys (main keyboard)
                    for i in range(min(10, n_actions)):
                        if (pygame.K_0 + i) in pressed_keys:
                            action_detected = i
                            break

                    # Check numpad keys if no main key pressed
                    if action_detected is None:
                        for i in range(min(10, n_actions)):
                            if (pygame.K_KP0 + i) in pressed_keys:
                                action_detected = i
                                break

                    # Use detected action, or default to 0 (NOOP-like)
                    action = action_detected if action_detected is not None else 0

                    # Update control window display to show current action
                    if pygame_screen:
                        font = pygame.font.Font(None, 24)
                        pygame_screen.fill((40, 40, 40))
                        text1 = font.render(f"Press keys 0-{n_actions-1} to select action", True, (255, 255, 255))
                        text2 = font.render("Press Q to quit", True, (255, 255, 255))
                        if action_detected is not None:
                            text3 = font.render(f"Current action: {action} (ACTIVE)", True, (100, 255, 100))
                        else:
                            text3 = font.render(f"Current action: {action} (default)", True, (180, 180, 180))
                        pygame_screen.blit(text1, (10, 15))
                        pygame_screen.blit(text2, (10, 45))
                        pygame_screen.blit(text3, (10, 75))
                        pygame.display.flip()
            else:
                # Fallback: blocking terminal input
                try:
                    user_input = input()
                except EOFError:
                    user_input = ""

                if user_input.strip().lower() in {"q", "quit", "exit"}:
                    break

                try:
                    action = int(user_input.strip()[0]) if user_input.strip() else 0
                    action = max(0, min(action, n_actions - 1))
                except (ValueError, IndexError):
                    action = 0

        # Execute action
        step_start = time.perf_counter()

        # Wrap action appropriately for vectorized env
        if is_multibinary:
            # MultiBinary: action is already an array, wrap in batch dimension
            action_batch = action.reshape(1, -1)
        else:
            # Discrete: single integer, wrap in array
            action_batch = np.array([action])

        obs, reward, terminated, truncated, info = env.step(action_batch)
        step_reward = reward[0]
        episode_reward += step_reward
        episode_length += 1

        # Update plotter with step reward
        if plotter and plotter.is_open:
            plotter.add_step(step_reward)

        # Apply FPS limiting if specified
        if frame_delay > 0:
            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Check if episode finished
        done = terminated[0] or truncated[0]
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            reported_episodes += 1

            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            print(
                f"[episodes {reported_episodes}/{target_episodes}] last_rew={episode_reward:.2f} last_len={episode_length} "
                f"mean_rew={mean_reward:.2f} mean_len={int(mean_length)}"
            )

            # Reset plotter for next episode
            if plotter and plotter.is_open:
                plotter.reset_episode()

            # Reset for next episode
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0

    # Cleanup pygame if it was initialized
    if pygame_available and pygame_screen:
        import pygame
        pygame.quit()


def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Play a trained agent using RolloutCollector (human render)")
    p.add_argument("id", nargs="?", default=None, help="Config ID (env:variant) or Run ID (auto-detected by presence of ':')")
    id_group = p.add_mutually_exclusive_group(required=False)
    id_group.add_argument("--run-id", default=None, help="Run ID under runs/ (default: last run with best checkpoint)")
    id_group.add_argument("--config-id", default=None, help="Config ID in 'env:variant' format (e.g., 'VizDoom-Basic-v0:ppo') - runs with random/user policy")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions (mode/argmax)")
    p.add_argument("--headless", action="store_true", default=False, help="Do not render the environment")
    p.add_argument(
        "--step-by-step",
        dest="step_by_step",
        action="store_true",
        help="Pause for input each env step for visual debugging",
    )
    p.add_argument(
        "--mode",
        choices=["trained", "random", "user"],
        default=None,
        help="Action mode: 'trained' (use trained policy), 'random' (sample from action space), 'user' (keyboard input). Default: 'trained' for --run-id, 'random' for --config-id",
    )
    p.add_argument("--seed", type=str, default=None, help="Random seed for environment (int, 'train', 'val', 'test', or None for test seed)")
    p.add_argument("--fps", type=int, default=None, help="Limit playback to target FPS (frames per second)")
    p.add_argument("--plot-rewards", dest="plot_rewards", action="store_true", default=True, help="Show real-time reward plot (default: True)")
    p.add_argument("--no-plot-rewards", dest="plot_rewards", action="store_false", help="Disable real-time reward plot")
    p.add_argument("--plot-update-interval", type=float, default=0.2, help="Minimum time (seconds) between plot updates (default: 0.2, lower=smoother but slower game)")
    args = p.parse_args()
    target_episodes = max(1, int(args.episodes))

    # Auto-detect config-id vs run-id from positional argument
    if args.id is not None:
        # Ensure no explicit --run-id or --config-id was also provided
        if args.run_id is not None or args.config_id is not None:
            raise ValueError("Cannot specify positional ID and --run-id/--config-id simultaneously")

        # Auto-detect: if contains ':', treat as config-id, otherwise as run-id
        if ':' in args.id:
            args.config_id = args.id
        else:
            args.run_id = args.id

    # Default to run-id mode if neither specified
    if args.run_id is None and args.config_id is None:
        args.run_id = "@last"

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Position game window on the left side to not overlap with plot window
    # SDL_VIDEO_WINDOW_POS sets the initial window position
    if not args.headless:
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "50,50")  # x=50 (left side), y=50 (near top)

    # Branch: config-id mode or run-id mode
    if args.config_id is not None:
        # Config-id mode: load config from YAML, no trained policy
        from utils.config import load_config

        # Parse config_id in 'env:variant' format
        if ':' not in args.config_id:
            raise ValueError(f"config_id must be in 'env:variant' format (e.g., 'VizDoom-Basic-v0:ppo'), got: {args.config_id}")
        env_id, variant_id = args.config_id.split(':', 1)

        # Load config
        config = load_config(env_id, variant_id)

        # Default mode to random for config-id
        if args.mode is None:
            args.mode = "random"

        # Trained mode is not allowed with config-id
        if args.mode == "trained":
            raise ValueError("--mode trained requires --run-id, not --config-id")

        run = None
    else:
        # Run-id mode: load trained policy from checkpoint
        # Resolve run ID (handle @last symlink)
        run_id = args.run_id
        if run_id == "@last":
            from utils.run import LAST_RUN_DIR
            if not LAST_RUN_DIR.exists():
                raise FileNotFoundError("No @last run found. Train a model first.")
            run_id = LAST_RUN_DIR.resolve().name

        # Check if run exists locally, if not try to download from W&B
        run_dir = Run._resolve_run_dir(run_id)
        if not run_dir.exists():
            print(f"Run {run_id} not found locally. Attempting to download from W&B...")
            from utils.wandb_artifacts import download_run_artifact
            download_run_artifact(run_id)

        # Load run and config
        run = Run.load(run_id)
        config = run.load_config()

        # Default mode to trained for run-id
        if args.mode is None:
            args.mode = "trained"

        # For trained mode, require checkpoint
        if args.mode == "trained":
            assert run.best_checkpoint_dir is not None, "run has no best checkpoint"

    # Fall back to render_fps from config spec if fps not provided
    if args.fps is None and config.spec and 'render_fps' in config.spec:
        args.fps = config.spec['render_fps']

    # Resolve seed argument
    if args.seed is None:
        # Default to test seed
        seed = config.seed_test
    elif args.seed in ["train", "val", "test"]:
        # Map stage names to corresponding seeds
        seeds = {
            "train": config.seed_train,
            "val": config.seed_val,
            "test": config.seed_test,
        }
        seed = seeds[args.seed]
    else:
        # Parse as integer
        seed = int(args.seed)

    # Seed all RNGs (Python, NumPy, PyTorch) for reproducibility
    set_random_seed(seed)

    # Build a single-env environment with human rendering
    # Force vectorization_mode='sync' to ensure render() is supported (ALE atari vectorization doesn't support it)
    env_overrides = {
        'n_envs': 1,
        'vectorization_mode': 'sync',
        'render_mode': "human" if not args.headless else None,
        'seed': seed
    }
    env = build_env_from_config(config, **env_overrides)

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    from gym_wrappers.vec_obs_printer import VecObsBarPrinter
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=target_episodes)

    # Extract action labels from config
    action_labels = extract_action_labels_from_config(config)

    # Initialize reward plotter if enabled and not headless
    plotter = None
    if args.plot_rewards and not args.headless:
        try:
            plotter = RewardPlotter(update_interval=args.plot_update_interval)
            backend = "PyQtGraph (fast)" if plotter.use_pyqtgraph else "Matplotlib"
            print(f"Real-time reward plot enabled using {backend} (update interval: {args.plot_update_interval}s).")
            print("Close the plot window to disable plotting.")
        except Exception as e:
            print(f"Warning: Could not initialize reward plotter: {e}")
            plotter = None

    # Handle different modes
    if args.mode in ["random", "user"]:
        # Manual control modes don't need policy
        try:
            play_episodes_manual(env, target_episodes, args.mode, args.step_by_step, args.fps, action_labels, plotter)
        finally:
            if plotter:
                plotter.close()
        print("Done.")
        return

    # Trained mode: load policy
    assert run is not None, "run must be loaded for trained mode"
    # TODO: we should be loading the agent and having it run the episode
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

    # Initialize rollout collector; step-by-step mode, FPS limiting, or plotting uses single-step rollouts
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1 if (args.step_by_step or args.fps or plotter) else config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    if args.step_by_step:
        print(f"Playing {target_episodes} episode(s) step-by-step with render_mode='human'...")
        print("Press Enter to step, 'q' then Enter to quit.")
    else:
        print(f"Playing {target_episodes} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    reported_episodes = 0
    frame_delay = 1.0 / args.fps if args.fps else 0
    try:
        while reported_episodes < target_episodes:
            if args.step_by_step:
                try:
                    user = input("")
                except EOFError:
                    user = ""
                if isinstance(user, str) and user.strip().lower() in {"q", "quit", "exit"}:
                    break

            step_start = time.perf_counter()
            _ = collector.collect(deterministic=args.deterministic)

            # Update plotter with step reward if plotting is enabled
            if plotter and plotter.is_open and collector.n_steps == 1:
                # When n_steps=1, buffer has exactly one step at the most recent position
                # Access the buffer's rewards_buf directly (shape: [n_steps, n_envs])
                step_reward = collector._buffer.rewards_buf[collector._buffer.pos - 1, 0]
                plotter.add_step(float(step_reward))

            # Apply FPS limiting if specified
            if frame_delay > 0:
                elapsed = time.perf_counter() - step_start
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            finished_eps = collector.pop_recent_episodes()
            if not finished_eps:
                continue  # Keep collecting until we finish a full episode

            for _env_idx, ep_rew, _ep_len, _was_timeout in finished_eps:
                if reported_episodes >= target_episodes:
                    break

                reported_episodes += 1
                metrics = collector.get_metrics()
                mean_rew = metrics.get('roll/ep_rew/mean', 0)
                last_len = int(_ep_len)
                mean_len = metrics.get('roll/ep_len/mean', 0)
                fps = metrics.get('roll/fps', 0)
                print(
                    f"[episodes {reported_episodes}/{target_episodes}] last_rew={ep_rew:.2f} last_len={last_len} "
                    f"mean_rew={mean_rew:.2f} mean_len={int(mean_len)} fps={fps:.1f}"
                )

                # Reset plotter for next episode
                if plotter and plotter.is_open:
                    plotter.reset_episode()
    finally:
        if plotter:
            plotter.close()

    print("Done.")


if __name__ == "__main__":
    main()
