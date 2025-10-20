#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.random import set_random_seed
from utils.rollouts import RolloutCollector
from utils.run import Run

# Global window layout file
WINDOW_LAYOUT_FILE = Path(__file__).parent / "window_layout.json"

# Global registry of active viewers for hotkey access
_active_viewers = {
    'reward_plotter': None,
    'observation_viewer': None,
    'cnn_filter_viewer': None
}


def save_window_layout():
    """Save current window positions to JSON file."""
    layout = {}

    for name, viewer in _active_viewers.items():
        if viewer is None:
            continue

        try:
            if hasattr(viewer, 'win') and hasattr(viewer.win, 'pos') and hasattr(viewer.win, 'size'):
                pos = viewer.win.pos()
                size = viewer.win.size()
                layout[name] = {
                    'x': pos.x(),
                    'y': pos.y(),
                    'width': size.width(),
                    'height': size.height()
                }
        except Exception:
            pass

    if layout:
        with open(WINDOW_LAYOUT_FILE, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"\n✓ Window layout saved to {WINDOW_LAYOUT_FILE}")
        print("  Positions:")
        for name, pos in layout.items():
            print(f"    {name}: ({pos['x']}, {pos['y']}) - {pos['width']}x{pos['height']}")
    else:
        print("\n✗ No windows to save")


def load_window_layout():
    """Load window positions from JSON file."""
    if not WINDOW_LAYOUT_FILE.exists():
        return {}

    try:
        with open(WINDOW_LAYOUT_FILE, 'r') as f:
            layout = json.load(f)
        # Filter out the 'note' field if present
        return {k: v for k, v in layout.items() if isinstance(v, dict) and 'x' in v}
    except Exception as e:
        print(f"Warning: Could not load window layout: {e}")
        return {}


def install_keyboard_shortcuts(win):
    """Install keyboard shortcuts on a PyQtGraph window."""
    try:
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        class KeyPressFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    # Check for F9 key (less likely to conflict)
                    if event.key() == QtCore.Qt.Key.Key_F9:
                        try:
                            # Use QTimer to defer save to avoid blocking the event loop
                            QtCore.QTimer.singleShot(0, save_window_layout)
                        except Exception as e:
                            print(f"Error in hotkey handler: {e}")
                        return False  # Don't consume the event to prevent freezing
                return False  # Don't consume other events

        # Create and install event filter
        key_filter = KeyPressFilter(win)
        win.installEventFilter(key_filter)
        # Store reference to prevent garbage collection
        win._key_filter = key_filter

    except Exception as e:
        print(f"Warning: Could not install keyboard shortcuts: {e}")


class RewardPlotter:
    """Real-time reward plotter for visualizing rewards during playback."""

    def __init__(self, update_interval: float = 0.2, max_steps_shown: int = 100):
        """Initialize the plotter with two subplots for episode and step rewards.

        Args:
            update_interval: Minimum time (in seconds) between plot updates to throttle rendering
            max_steps_shown: Maximum number of steps to show in sliding window (default: 100)
        """
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore
            import time

            self.pg = pg
            self.time = time
            self.max_steps_shown = max_steps_shown

            # Set background to white for better visibility
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

            # Create window with two plots (don't show yet)
            self.win = pg.GraphicsLayoutWidget(show=False, title="Real-time Reward Visualization")
            self.win.resize(500, 400)
            self.win.setWindowTitle('Real-time Reward Visualization')

            # Set window flags before showing
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'reward_plotter' in layout:
                pos = layout['reward_plotter']
                self.win.move(pos['x'], pos['y'])
                if pos['width'] and pos['height']:
                    self.win.resize(pos['width'], pos['height'])
            else:
                # Default position: right of the game window
                self.win.move(750, 50)

            # Episode reward plot (accumulated reward over steps)
            self.plot_episode = self.win.addPlot(title="<span style='font-size: 9pt'>Accumulated Reward per Episode</span>")
            self.plot_episode.setLabel('left', 'Accumulated Episode Reward', **{'font-size': '8pt'})
            self.plot_episode.setLabel('bottom', 'Step', **{'font-size': '8pt'})
            self.plot_episode.showGrid(x=True, y=True, alpha=0.3)
            self.plot_episode.addLegend(offset=(5, 5), labelTextSize='7pt')
            # Set smaller font size for tick labels
            font = pg.QtGui.QFont()
            font.setPointSize(7)
            self.plot_episode.getAxis('left').setTickFont(font)
            self.plot_episode.getAxis('bottom').setTickFont(font)
            self.plot_episode.getAxis('left').setStyle(tickTextOffset=3)
            self.plot_episode.getAxis('bottom').setStyle(tickTextOffset=3)
            # Disable auto-range on x-axis to maintain sliding window
            self.plot_episode.enableAutoRange(axis='x', enable=False)
            self.plot_episode.enableAutoRange(axis='y', enable=True)
            # Set initial x-range
            self.plot_episode.setXRange(0, max_steps_shown, padding=0)

            # Move to next row
            self.win.nextRow()

            # Step reward plot (individual step rewards)
            self.plot_step = self.win.addPlot(title="<span style='font-size: 9pt'>Individual Step Rewards</span>")
            self.plot_step.setLabel('left', 'Step Reward', **{'font-size': '8pt'})
            self.plot_step.setLabel('bottom', 'Step', **{'font-size': '8pt'})
            self.plot_step.showGrid(x=True, y=True, alpha=0.3)
            # Set smaller font size for tick labels
            self.plot_step.getAxis('left').setTickFont(font)
            self.plot_step.getAxis('bottom').setTickFont(font)
            self.plot_step.getAxis('left').setStyle(tickTextOffset=3)
            self.plot_step.getAxis('bottom').setStyle(tickTextOffset=3)
            # Disable auto-range on x-axis to maintain sliding window
            self.plot_step.enableAutoRange(axis='x', enable=False)
            self.plot_step.enableAutoRange(axis='y', enable=True)
            # Set initial x-range
            self.plot_step.setXRange(0, max_steps_shown, padding=0)

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

            # Connect window close event
            self.win.closeEvent = lambda event: setattr(self, 'is_open', False)

            # Throttling: only update plot at fixed intervals
            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            self.use_pyqtgraph = True

            # Install keyboard shortcuts (Ctrl+S to save layout)
            install_keyboard_shortcuts(self.win)

            # Show window after all setup is complete
            self.win.show()
            self.win.raise_()

        except ImportError:
            # Fallback to matplotlib if pyqtgraph is not available
            import matplotlib.pyplot as plt
            import time

            self.plt = plt
            self.time = time
            self.max_steps_shown = max_steps_shown
            self.plt.ion()

            self.fig, (self.ax_episode, self.ax_step) = self.plt.subplots(2, 1, figsize=(5, 4))
            self.fig.suptitle('Real-time Reward Visualization', fontsize=9, fontweight='bold')

            # Position window to the right of the game window
            try:
                manager = self.plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    # Try to set position (works with TkAgg backend)
                    manager.window.wm_geometry("+750+50")
            except Exception:
                pass  # If positioning fails, continue anyway

            self.ax_episode.set_xlabel('Step', fontsize=7)
            self.ax_episode.set_ylabel('Accumulated Episode Reward', fontsize=7)
            self.ax_episode.set_title('Accumulated Reward per Episode', fontsize=8)
            self.ax_episode.grid(True, alpha=0.3)
            self.ax_episode.tick_params(labelsize=6)

            self.ax_step.set_xlabel('Step', fontsize=7)
            self.ax_step.set_ylabel('Step Reward', fontsize=7)
            self.ax_step.set_title('Individual Step Rewards', fontsize=8)
            self.ax_step.grid(True, alpha=0.3)
            self.ax_step.tick_params(labelsize=6)
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

        # Cleanup old data that's outside the window to prevent memory bloat
        # Keep a buffer of 500 steps before the window to handle edge cases
        if self.global_step > self.max_steps_shown + 500:
            window_start = self.global_step - self.max_steps_shown - 500

            # Trim current episode data if it's getting too long
            if len(self.current_episode_steps) > self.max_steps_shown + 500:
                # Find index where steps >= window_start
                trim_idx = 0
                for i, step in enumerate(self.current_episode_steps):
                    if step >= window_start:
                        trim_idx = i
                        break

                if trim_idx > 0:
                    # Trim the lists
                    self.current_episode_steps = self.current_episode_steps[trim_idx:]
                    self.current_episode_rewards = self.current_episode_rewards[trim_idx:]
                    self.current_step_rewards = self.current_step_rewards[trim_idx:]

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

        # Cleanup old episodes that are completely outside the sliding window
        # Keep episodes that have any steps >= (current_step - window_size - buffer)
        if self.global_step > self.max_steps_shown + 500:
            window_start = self.global_step - self.max_steps_shown - 500
            # Filter out episodes where the last step is before the window
            self.episode_data = [
                ep for ep in self.episode_data
                if ep[0] and ep[0][-1] >= window_start
            ]

        # Reset for next episode
        self.current_episode_steps = []
        self.current_episode_rewards = []
        self.current_step_rewards = []
        self.current_curve = None
        self.episode_num += 1

        # Force an update at episode boundaries (always, regardless of throttling)
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
                from pyqtgraph.Qt import QtCore

                # Calculate sliding window range (fixed width)
                # Keep window width constant even at the start
                if self.global_step < self.max_steps_shown:
                    window_start = 0
                    window_end = self.max_steps_shown
                else:
                    window_start = self.global_step - self.max_steps_shown
                    window_end = self.global_step

                # Clear both plots completely
                self.plot_episode.clear()
                self.plot_step.clear()

                # Re-add zero line to step plot
                self.plot_step.addLine(y=0, pen=self.pg.mkPen('k', width=1, style=QtCore.Qt.PenStyle.DashLine))

                # Plot completed episodes (only visible portion)
                for i, episode_tuple in enumerate(self.episode_data):
                    steps = episode_tuple[0]
                    rewards = episode_tuple[1]

                    # Check if episode overlaps with window
                    if steps and steps[-1] >= window_start:
                        # Filter episode data to window (both start AND end bounds)
                        steps_array = np.array(steps)
                        rewards_array = np.array(rewards)
                        mask = (steps_array >= window_start) & (steps_array < window_end)
                        visible_steps = steps_array[mask]
                        visible_rewards = rewards_array[mask]

                        if len(visible_steps) > 0:
                            color = self.colors[i % len(self.colors)]
                            self.plot_episode.plot(
                                visible_steps, visible_rewards,
                                pen=self.pg.mkPen(color=color, width=2, style=QtCore.Qt.PenStyle.DashLine),
                                name=f'Episode {i+1}'
                            )

                # Plot current episode (only visible portion)
                if self.current_episode_steps:
                    steps_array = np.array(self.current_episode_steps)
                    rewards_array = np.array(self.current_episode_rewards)
                    step_rewards_array = np.array(self.current_step_rewards)

                    # Filter to window (both start AND end bounds)
                    mask = (steps_array >= window_start) & (steps_array < window_end)
                    visible_steps = steps_array[mask]
                    visible_rewards = rewards_array[mask]
                    visible_step_rewards = step_rewards_array[mask]

                    if len(visible_steps) > 0:
                        # Plot episode reward curve
                        color = self.colors[self.episode_num % len(self.colors)]
                        self.plot_episode.plot(
                            visible_steps,
                            visible_rewards,
                            pen=self.pg.mkPen(color=color, width=2),
                            name=f'Episode {self.episode_num+1}'
                        )

                        # Plot step rewards as bars
                        if len(visible_step_rewards) > 0:
                            bars = self.pg.BarGraphItem(
                                x=visible_steps, height=visible_step_rewards, width=0.8,
                                brush=self.pg.mkBrush(70, 130, 180, 150)
                            )
                            self.plot_step.addItem(bars)

                # Set x-axis range to the sliding window (padding=0 to avoid auto-scaling)
                self.plot_episode.setXRange(window_start, window_end, padding=0)
                self.plot_step.setXRange(window_start, window_end, padding=0)

            else:
                # Matplotlib (fallback) implementation with sliding window
                import numpy as np

                # Calculate sliding window range (fixed width)
                # Keep window width constant even at the start
                if self.global_step < self.max_steps_shown:
                    window_start = 0
                    window_end = self.max_steps_shown
                else:
                    window_start = self.global_step - self.max_steps_shown
                    window_end = self.global_step

                # Clear plots
                self.ax_episode.clear()
                self.ax_step.clear()

                # Reapply labels and formatting
                self.ax_episode.set_xlabel('Step', fontsize=7)
                self.ax_episode.set_ylabel('Accumulated Episode Reward', fontsize=7)
                self.ax_episode.set_title('Accumulated Reward per Episode', fontsize=8)
                self.ax_episode.grid(True, alpha=0.3)
                self.ax_episode.tick_params(labelsize=6)

                self.ax_step.set_xlabel('Step', fontsize=7)
                self.ax_step.set_ylabel('Step Reward', fontsize=7)
                self.ax_step.set_title('Individual Step Rewards', fontsize=8)
                self.ax_step.grid(True, alpha=0.3)
                self.ax_step.tick_params(labelsize=6)
                self.ax_step.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

                # Plot completed episodes (filtered to window)
                for i, episode_tuple in enumerate(self.episode_data):
                    steps = episode_tuple[0]
                    rewards = episode_tuple[1]

                    # Check if episode overlaps with window
                    if steps and steps[-1] >= window_start:
                        steps_array = np.array(steps)
                        rewards_array = np.array(rewards)
                        mask = (steps_array >= window_start) & (steps_array < window_end)
                        visible_steps = steps_array[mask]
                        visible_rewards = rewards_array[mask]

                        if len(visible_steps) > 0:
                            color = self.colors[i % len(self.colors)]
                            self.ax_episode.plot(visible_steps, visible_rewards, color=color,
                                               linewidth=2, alpha=0.7, label=f'Episode {i+1}')

                # Plot current episode (filtered to window)
                if self.current_episode_steps:
                    steps_array = np.array(self.current_episode_steps)
                    rewards_array = np.array(self.current_episode_rewards)
                    mask = (steps_array >= window_start) & (steps_array < window_end)
                    visible_steps = steps_array[mask]
                    visible_rewards = rewards_array[mask]

                    if len(visible_steps) > 0:
                        color = self.colors[self.episode_num % len(self.colors)]
                        self.ax_episode.plot(visible_steps, visible_rewards,
                                            color=color, linewidth=2, label=f'Episode {self.episode_num+1} (current)')

                # Plot step rewards (filtered to window)
                if self.current_step_rewards:
                    steps_array = np.array(self.current_episode_steps)
                    rewards_array = np.array(self.current_step_rewards)
                    mask = (steps_array >= window_start) & (steps_array < window_end)
                    visible_steps = steps_array[mask]
                    visible_step_rewards = rewards_array[mask]

                    if len(visible_steps) > 0:
                        self.ax_step.bar(visible_steps, visible_step_rewards, width=1.0, alpha=0.6, color='steelblue')

                # Set x-axis limits to sliding window
                self.ax_episode.set_xlim(window_start, window_end)
                self.ax_step.set_xlim(window_start, window_end)

                # Add legends
                if len(self.episode_data) + (1 if self.current_episode_steps else 0) > 0:
                    self.ax_episode.legend(loc='best', fontsize=6)

                self.plt.tight_layout()
                # Draw the canvas and flush events without blocking
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        except Exception as e:
            # Log error and mark plotter as closed to prevent further crashes
            import sys
            import traceback
            print(f"Error updating reward plot: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
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


class CNNFilterActivationDetailViewer:
    """Detail viewer for a single filter/activation pair."""

    def __init__(self, layer_idx, filter_idx, filter_data, activation_data=None):
        """Initialize detail viewer for a single filter/activation.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index within the layer
            filter_data: Filter kernel data (2D numpy array)
            activation_data: Activation map data (2D numpy array, optional)
        """
        try:
            import pyqtgraph as pg
            import numpy as np

            self.pg = pg
            self.np = np
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx

            # Set background to black
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')

            # Create window
            title = f"Layer {layer_idx} Filter {filter_idx}"
            self.win = pg.GraphicsLayoutWidget(show=False, title=title)
            self.win.setWindowTitle(title)

            # Set window flags
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Remove spacing
            self.win.ci.layout.setSpacing(0)
            self.win.ci.layout.setContentsMargins(0, 0, 0, 0)

            # Create views for filter and activation
            # Filter view
            filter_view = self.win.addViewBox()
            filter_view.setAspectLocked(True)
            filter_view.invertY(True)
            filter_view.setContentsMargins(0, 0, 0, 0)
            filter_view.setMenuEnabled(False)
            filter_view.setMouseEnabled(x=False, y=False)
            filter_view.enableAutoRange(enable=False)
            filter_item = pg.ImageItem()
            filter_view.addItem(filter_item)

            # Activation view (if activation data provided)
            if activation_data is not None:
                act_view = self.win.addViewBox()
                act_view.setAspectLocked(True)
                act_view.invertY(True)
                act_view.setContentsMargins(0, 0, 0, 0)
                act_view.setMenuEnabled(False)
                act_view.setMouseEnabled(x=False, y=False)
                act_view.enableAutoRange(enable=False)
                act_item = pg.ImageItem()
                act_view.addItem(act_item)
            else:
                act_view = None
                act_item = None

            self.filter_view = filter_view
            self.filter_item = filter_item
            self.act_view = act_view
            self.act_item = act_item

            # Render the data
            self._render(filter_data, activation_data)

            self.is_open = True
            self.win.closeEvent = lambda event: setattr(self, 'is_open', False)

            # Show window
            self.win.show()
            self.win.raise_()

        except ImportError:
            raise ImportError("PyQtGraph is required for detail viewer")

    def _render(self, filter_data, activation_data):
        """Render filter and activation data."""
        # Normalize filter to 0-255
        f_min, f_max = filter_data.min(), filter_data.max()
        if f_max > f_min:
            filter_norm = ((filter_data - f_min) / (f_max - f_min) * 255).astype(self.np.uint8)
        else:
            filter_norm = self.np.zeros_like(filter_data, dtype=self.np.uint8)

        # Create RGB image for filter (grayscale)
        fh, fw = filter_data.shape
        filter_rgb = self.np.zeros((fh, fw, 3), dtype=self.np.uint8)
        filter_rgb[:, :, 0] = filter_norm
        filter_rgb[:, :, 1] = filter_norm
        filter_rgb[:, :, 2] = filter_norm

        # Transpose for PyQtGraph
        filter_rgb_t = self.np.transpose(filter_rgb, (1, 0, 2))
        self.filter_item.setImage(filter_rgb_t)
        self.filter_view.setRange(xRange=(0, fh), yRange=(0, fw), padding=0)

        # Size window based on filter size (scale up small filters)
        scale = max(1, 200 // max(fh, fw))
        window_width = fh * scale
        window_height = fw * scale

        # Render activation if provided
        if activation_data is not None and self.act_item is not None:
            # Normalize activation to 0-255
            a_min, a_max = activation_data.min(), activation_data.max()
            if a_max > a_min:
                act_norm = ((activation_data - a_min) / (a_max - a_min) * 255).astype(self.np.uint8)
            else:
                act_norm = self.np.zeros_like(activation_data, dtype=self.np.uint8)

            # Create RGB image for activation (grayscale)
            ah, aw = activation_data.shape
            act_rgb = self.np.zeros((ah, aw, 3), dtype=self.np.uint8)
            act_rgb[:, :, 0] = act_norm
            act_rgb[:, :, 1] = act_norm
            act_rgb[:, :, 2] = act_norm

            # Transpose for PyQtGraph
            act_rgb_t = self.np.transpose(act_rgb, (1, 0, 2))
            self.act_item.setImage(act_rgb_t)
            self.act_view.setRange(xRange=(0, ah), yRange=(0, aw), padding=0)

            # Add activation width to window
            window_width += ah * scale

        self.win.resize(int(window_width), int(window_height) + 30)

    def update_filter(self, layer_idx, filter_idx, filter_data, activation_data=None):
        """Update the viewer with a new filter/activation pair."""
        if not self.is_open:
            return

        self.layer_idx = layer_idx
        self.filter_idx = filter_idx

        # Update window title
        self.win.setWindowTitle(f"Layer {layer_idx} Filter {filter_idx}")

        # Re-render with new data
        self._render(filter_data, activation_data)

    def update_activation(self, activation_data):
        """Update the activation display with new data."""
        if not self.is_open or self.act_item is None or activation_data is None:
            return

        # Normalize activation to 0-255
        a_min, a_max = activation_data.min(), activation_data.max()
        if a_max > a_min:
            act_norm = ((activation_data - a_min) / (a_max - a_min) * 255).astype(self.np.uint8)
        else:
            act_norm = self.np.zeros_like(activation_data, dtype=self.np.uint8)

        # Create RGB image for activation (grayscale)
        ah, aw = activation_data.shape
        act_rgb = self.np.zeros((ah, aw, 3), dtype=self.np.uint8)
        act_rgb[:, :, 0] = act_norm
        act_rgb[:, :, 1] = act_norm
        act_rgb[:, :, 2] = act_norm

        # Transpose for PyQtGraph
        act_rgb_t = self.np.transpose(act_rgb, (1, 0, 2))
        self.act_item.setImage(act_rgb_t)

    def close(self):
        """Close the detail viewer."""
        if self.is_open:
            try:
                self.win.close()
            except Exception:
                pass
            self.is_open = False


class CNNFilterActivationViewer:
    """Real-time CNN filter and activation visualizer."""

    def __init__(self, policy_model, update_interval: float = 0.1):
        """Initialize the viewer for CNN filters and activations.

        Args:
            policy_model: The policy model (must have a 'cnn' attribute)
            update_interval: Minimum time (in seconds) between updates to throttle rendering
        """
        import torch.nn as nn
        import time

        self.time = time
        self.update_interval = update_interval
        self.last_update_time = time.time()
        self.needs_update = False
        self.is_open = False  # Initialize early, before hooks
        self.use_pyqtgraph = False  # Initialize early
        self._window_sized = False  # Initialize early

        # Extract Conv2d layers from the CNN
        if not hasattr(policy_model, 'cnn'):
            raise ValueError("Policy model must have a 'cnn' attribute for filter visualization")

        self.conv_layers = []
        self.conv_info = []
        for i, module in enumerate(policy_model.cnn):
            if isinstance(module, nn.Conv2d):
                self.conv_layers.append(module)
                self.conv_info.append({
                    'index': i,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'activation': None,  # Will be populated by hooks
                    'filter_weights': None,  # Will store weights for click handling
                    'grid_layout': None,  # Will store grid layout info for click detection
                })

        if not self.conv_layers:
            raise ValueError("No Conv2d layers found in policy model")

        # Track single detail viewer (reused for all clicks)
        self.detail_viewer = None

        # Register forward hooks to capture activations
        self.activation_handles = []
        for idx, layer in enumerate(self.conv_layers):
            handle = layer.register_forward_hook(self._make_activation_hook(idx))
            self.activation_handles.append(handle)

        # Store filter weights for click handling
        import torch
        for idx, layer in enumerate(self.conv_layers):
            with torch.no_grad():
                self.conv_info[idx]['filter_weights'] = layer.weight.detach().cpu().numpy()

        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtWidgets
            import numpy as np

            self.pg = pg
            self.np = np

            # Set background to black
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')

            # Create window with grid layout (don't show yet)
            self.win = pg.GraphicsLayoutWidget(show=False, title="CNN Filters & Activations")
            # Start with small size, will resize after content is rendered
            self.win.resize(400, 200)
            self.win.setWindowTitle('CNN Filters & Activations')

            # Set window flags before showing
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'cnn_filter_viewer' in layout:
                pos = layout['cnn_filter_viewer']
                self.win.move(pos['x'], pos['y'])
                # Don't apply width/height here as it will be auto-sized
            else:
                # Default position: below the reward plot
                self.win.move(750, 500)

            # Remove all spacing and margins for tight layout
            self.win.ci.layout.setSpacing(0)
            self.win.ci.layout.setContentsMargins(0, 0, 0, 0)

            # Create image items for each layer (filters + activations)
            self.filter_items = []
            self.activation_items = []
            self.filter_views = []
            self.activation_views = []

            for idx, info in enumerate(self.conv_info):
                # Add row for this layer
                label = f"Layer {idx} ({info['out_channels']} filters, {info['kernel_size']}x{info['kernel_size']})"

                # Filters column
                filter_view = self.win.addViewBox()
                filter_view.setAspectLocked(True)
                filter_view.invertY(True)
                filter_view.setContentsMargins(0, 0, 0, 0)
                filter_view.setMenuEnabled(False)
                filter_view.setMouseEnabled(x=False, y=False)
                filter_view.enableAutoRange(enable=False)
                filter_item = pg.ImageItem()
                filter_view.addItem(filter_item)

                # Connect click handler to the scene
                filter_view.scene().sigMouseClicked.connect(self._make_filter_scene_click_handler(idx, filter_view))

                self.filter_items.append(filter_item)
                self.filter_views.append(filter_view)

                # Activations column
                act_view = self.win.addViewBox()
                act_view.setAspectLocked(True)
                act_view.invertY(True)
                act_view.setContentsMargins(0, 0, 0, 0)
                act_view.setMenuEnabled(False)
                act_view.setMouseEnabled(x=False, y=False)
                act_view.enableAutoRange(enable=False)
                act_item = pg.ImageItem()
                act_view.addItem(act_item)

                # Connect click handler to the scene
                act_view.scene().sigMouseClicked.connect(self._make_activation_scene_click_handler(idx, act_view))

                self.activation_items.append(act_item)
                self.activation_views.append(act_view)

                # Move to next row
                self.win.nextRow()

            self.is_open = True
            self.use_pyqtgraph = True
            self._window_sized = False

            # Render filters once (they don't change)
            self._render_filters()

            # Install keyboard shortcuts (Ctrl+S to save layout)
            install_keyboard_shortcuts(self.win)

            # Show window after all setup is complete
            self.win.show()
            self.win.raise_()

        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            import numpy as np

            self.plt = plt
            self.np = np

            self.plt.ion()
            n_layers = len(self.conv_layers)
            self.fig, self.axes = self.plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))
            if n_layers == 1:
                self.axes = self.axes.reshape(1, -1)

            self.fig.suptitle('CNN Filters & Activations', fontsize=14, fontweight='bold')

            # Position window below the reward plot
            try:
                manager = self.plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    # Try to set position (works with TkAgg backend)
                    manager.window.wm_geometry("+750+500")
            except Exception:
                pass  # If positioning fails, continue anyway

            for idx, info in enumerate(self.conv_info):
                self.axes[idx, 0].set_title(f"Layer {idx} Filters", fontsize=10)
                self.axes[idx, 1].set_title(f"Layer {idx} Activations", fontsize=10)
                self.axes[idx, 0].axis('off')
                self.axes[idx, 1].axis('off')

            self.is_open = True
            self.use_pyqtgraph = False

            # Tight layout with minimal spacing
            self.plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
            self.plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Render filters once
            self._render_filters()

    def _make_activation_hook(self, layer_idx):
        """Create a forward hook to capture activations."""
        def hook(module, input, output):
            if not self.is_open:
                return
            # Store activation (detach and move to CPU)
            import torch
            self.conv_info[layer_idx]['activation'] = output.detach().cpu()
            self.needs_update = True
        return hook

    def _make_filter_scene_click_handler(self, layer_idx, view_box):
        """Create a scene click handler for filter images."""
        def handler(event):
            # Check if click is on this specific ViewBox
            scene_pos = event.scenePos()
            if view_box.sceneBoundingRect().contains(scene_pos):
                # Map scene coordinates to view coordinates
                view_pos = view_box.mapSceneToView(scene_pos)
                filter_idx = self._detect_filter_from_click(layer_idx, view_pos.x(), view_pos.y())
                if filter_idx is not None:
                    self._open_detail_viewer(layer_idx, filter_idx)
                    event.accept()
        return handler

    def _make_activation_scene_click_handler(self, layer_idx, view_box):
        """Create a scene click handler for activation images."""
        def handler(event):
            # Check if click is on this specific ViewBox
            scene_pos = event.scenePos()
            if view_box.sceneBoundingRect().contains(scene_pos):
                # Map scene coordinates to view coordinates
                view_pos = view_box.mapSceneToView(scene_pos)
                filter_idx = self._detect_activation_from_click(layer_idx, view_pos.x(), view_pos.y())
                if filter_idx is not None:
                    self._open_detail_viewer(layer_idx, filter_idx)
                    event.accept()
        return handler

    def _detect_filter_from_click(self, layer_idx, x, y):
        """Detect which filter was clicked based on coordinates.

        Note: x, y are in transposed coordinate system (W, H) due to PyQtGraph transpose.
        """
        info = self.conv_info[layer_idx]
        layout = info.get('grid_layout')
        if layout is None:
            return None

        # Swap x, y back to (H, W) coordinate system
        y, x = x, y

        grid_rows = layout['grid_rows']
        grid_cols = layout['grid_cols']
        kH = layout['kH']
        kW = layout['kW']
        padding = layout['padding']

        # Calculate which grid cell was clicked
        # Account for padding borders
        if x < padding or y < padding:
            return None

        # Remove the initial padding offset
        x_in_grid = x - padding
        y_in_grid = y - padding

        # Calculate cell position
        cell_width = kW + padding
        cell_height = kH + padding

        col = int(x_in_grid / cell_width)
        row = int(y_in_grid / cell_height)

        # Check if click is within valid grid bounds
        if row >= grid_rows or col >= grid_cols:
            return None

        # Check if click is on border (padding area within cell)
        x_in_cell = x_in_grid % cell_width
        y_in_cell = y_in_grid % cell_height

        # If click is on the right or bottom border of the cell, ignore
        if x_in_cell >= kW or y_in_cell >= kH:
            return None

        # Calculate filter index
        filter_idx = row * grid_cols + col

        # Check if filter index is valid
        if filter_idx >= info['out_channels']:
            return None

        return filter_idx

    def _detect_activation_from_click(self, layer_idx, x, y):
        """Detect which activation was clicked based on coordinates.

        Note: x, y are in transposed coordinate system (W, H) due to PyQtGraph transpose.
        """
        info = self.conv_info[layer_idx]
        layout = info.get('activation_layout')
        if layout is None:
            return None

        # Swap x, y back to (H, W) coordinate system
        y, x = x, y

        grid_rows = layout['grid_rows']
        grid_cols = layout['grid_cols']
        H = layout['H']
        W = layout['W']
        padding = layout['padding']

        # Calculate which grid cell was clicked
        # Account for padding borders
        if x < padding or y < padding:
            return None

        # Remove the initial padding offset
        x_in_grid = x - padding
        y_in_grid = y - padding

        # Calculate cell position
        cell_width = W + padding
        cell_height = H + padding

        col = int(x_in_grid / cell_width)
        row = int(y_in_grid / cell_height)

        # Check if click is within valid grid bounds
        if row >= grid_rows or col >= grid_cols:
            return None

        # Check if click is on border (padding area within cell)
        x_in_cell = x_in_grid % cell_width
        y_in_cell = y_in_grid % cell_height

        # If click is on the right or bottom border of the cell, ignore
        if x_in_cell >= W or y_in_cell >= H:
            return None

        # Calculate activation index
        act_idx = row * grid_cols + col

        # Check if activation index is valid
        n_channels = layout['n_channels']
        if act_idx >= n_channels:
            return None

        return act_idx

    def _open_detail_viewer(self, layer_idx, filter_idx):
        """Open or update the detail viewer for the specified filter/activation pair."""
        info = self.conv_info[layer_idx]

        # Get filter data
        weights = info['filter_weights']
        out_ch, in_ch, kH, kW = weights.shape

        # Extract and process filter for this channel
        if in_ch == 1:
            filter_2d = weights[filter_idx, 0, :, :]
        elif in_ch == 3:
            # Convert RGB to grayscale
            filter_2d = (0.299 * weights[filter_idx, 0, :, :] +
                        0.587 * weights[filter_idx, 1, :, :] +
                        0.114 * weights[filter_idx, 2, :, :])
        else:
            # Average over input channels
            filter_2d = weights[filter_idx].mean(axis=0)

        # Get activation data if available
        activation = info.get('activation')
        activation_2d = None
        if activation is not None:
            # activation shape: (batch, channels, H, W)
            if activation.dim() == 4:
                activation = activation[0]  # Take first batch
            if filter_idx < activation.shape[0]:
                activation_2d = activation[filter_idx].cpu().numpy()

        # Create or update the detail viewer
        try:
            if self.detail_viewer is None or not self.detail_viewer.is_open:
                # Create new viewer on first click or if closed
                self.detail_viewer = CNNFilterActivationDetailViewer(
                    layer_idx, filter_idx, filter_2d, activation_2d
                )
                self.detail_viewer.win.move(800, 100)
                print(f"Opened detail viewer: Layer {layer_idx} Filter {filter_idx}")
            else:
                # Update existing viewer with new filter/activation
                self.detail_viewer.update_filter(layer_idx, filter_idx, filter_2d, activation_2d)
                self.detail_viewer.win.raise_()  # Bring to front
                print(f"Updated detail viewer: Layer {layer_idx} Filter {filter_idx}")
        except Exception as e:
            import sys
            print(f"Error with detail viewer: {e}", file=sys.stderr)

    def _render_filters(self):
        """Render filter kernels for all Conv2d layers."""
        import torch

        for idx, (layer, info) in enumerate(zip(self.conv_layers, self.conv_info)):
            # Get filter weights: shape (out_channels, in_channels, kH, kW)
            with torch.no_grad():
                weights = layer.weight.detach().cpu().numpy()

            # Create a grid visualization of filters (now returns RGB)
            filter_grid = self._create_filter_grid(weights, store_layout_idx=idx)

            # Store filter grid dimensions for window sizing (H, W, 3)
            info['filter_grid_shape'] = filter_grid.shape[:2]  # (H, W)

            # Display based on backend
            if self.use_pyqtgraph:
                # Filter grid is already 0-255 RGB, just convert to uint8
                filter_grid_norm = filter_grid.astype(self.np.uint8)
                # PyQtGraph expects (W, H, 3) format for RGB images, but numpy creates (H, W, 3)
                # Transpose spatial dimensions: (H, W, 3) -> (W, H, 3)
                filter_grid_transposed = self.np.transpose(filter_grid_norm, (1, 0, 2))
                self.filter_items[idx].setImage(filter_grid_transposed)

                # Set view range to match transposed dimensions
                h, w = filter_grid.shape[:2]
                self.filter_views[idx].setRange(xRange=(0, h), yRange=(0, w), padding=0)
            else:
                self.axes[idx, 0].clear()
                # Display RGB image (no colormap)
                filter_grid_norm = filter_grid.astype(self.np.uint8)
                self.axes[idx, 0].imshow(filter_grid_norm)
                self.axes[idx, 0].set_title(f"Layer {idx} Filters ({info['out_channels']} × {info['kernel_size']})", fontsize=9)
                self.axes[idx, 0].axis('off')

    def _create_filter_grid(self, weights, store_layout_idx=None):
        """Create a grid visualization of filter kernels.

        Args:
            weights: numpy array of shape (out_channels, in_channels, kH, kW)
            store_layout_idx: If provided, store grid layout in conv_info[store_layout_idx]

        Returns:
            RGB numpy array (H, W, 3) representing the grid of filters with black borders
        """
        out_ch, in_ch, kH, kW = weights.shape

        # For visualization, average over input channels or take first channel
        if in_ch == 1:
            filters_2d = weights[:, 0, :, :]  # (out_ch, kH, kW)
        elif in_ch == 3:
            # For RGB input, convert to grayscale using standard weights
            filters_2d = (0.299 * weights[:, 0, :, :] +
                         0.587 * weights[:, 1, :, :] +
                         0.114 * weights[:, 2, :, :])  # (out_ch, kH, kW)
        else:
            # Average over input channels
            filters_2d = weights.mean(axis=1)  # (out_ch, kH, kW)

        # Arrange filters in a grid (roughly square)
        import math
        grid_cols = int(math.ceil(math.sqrt(out_ch)))
        grid_rows = int(math.ceil(out_ch / grid_cols))

        # 1px padding for border
        padding = 1
        grid_h = grid_rows * kH + (grid_rows + 1) * padding  # +1 for borders on all sides
        grid_w = grid_cols * kW + (grid_cols + 1) * padding  # +1 for borders on all sides

        # Store layout info for click detection
        if store_layout_idx is not None:
            self.conv_info[store_layout_idx]['grid_layout'] = {
                'grid_rows': grid_rows,
                'grid_cols': grid_cols,
                'kH': kH,
                'kW': kW,
                'padding': padding,
            }

        # Create RGB grid (start with black border color: R=0, G=0, B=0)
        grid = self.np.zeros((grid_h, grid_w, 3), dtype=self.np.float32)
        # All channels stay 0 for black

        for i in range(out_ch):
            row = i // grid_cols
            col = i % grid_cols

            # Add padding offset to position each filter
            y_start = row * (kH + padding) + padding
            x_start = col * (kW + padding) + padding

            # Normalize filter to 0-255 range for each filter individually
            filt = filters_2d[i]
            f_min, f_max = filt.min(), filt.max()
            if f_max > f_min:
                filt_norm = (filt - f_min) / (f_max - f_min) * 255
            else:
                filt_norm = self.np.zeros_like(filt)

            # Place grayscale filter in all RGB channels (creates grayscale appearance)
            grid[y_start:y_start + kH, x_start:x_start + kW, 0] = filt_norm  # R
            grid[y_start:y_start + kH, x_start:x_start + kW, 1] = filt_norm  # G
            grid[y_start:y_start + kH, x_start:x_start + kW, 2] = filt_norm  # B

        return grid

    def _create_activation_grid(self, activations, store_layout_idx=None):
        """Create a grid visualization of activation maps.

        Args:
            activations: tensor of shape (batch, channels, H, W)
            store_layout_idx: If provided, store grid layout in conv_info[store_layout_idx]

        Returns:
            RGB numpy array (H, W, 3) representing the grid of activation maps with black borders
        """
        # Take first batch element
        if activations.dim() == 4:
            activations = activations[0]  # (channels, H, W)

        act_np = activations.cpu().numpy()
        n_channels, H, W = act_np.shape

        # Show a subset of channels if too many (max 64)
        max_channels = 64
        if n_channels > max_channels:
            # Sample evenly spaced channels
            indices = self.np.linspace(0, n_channels - 1, max_channels, dtype=int)
            act_np = act_np[indices]
            n_channels = max_channels

        # Arrange activation maps in a grid
        import math
        grid_cols = int(math.ceil(math.sqrt(n_channels)))
        grid_rows = int(math.ceil(n_channels / grid_cols))

        # 1px padding for border
        padding = 1
        grid_h = grid_rows * H + (grid_rows + 1) * padding  # +1 for borders on all sides
        grid_w = grid_cols * W + (grid_cols + 1) * padding  # +1 for borders on all sides

        # Store layout info for click detection
        if store_layout_idx is not None:
            self.conv_info[store_layout_idx]['activation_layout'] = {
                'grid_rows': grid_rows,
                'grid_cols': grid_cols,
                'H': H,
                'W': W,
                'padding': padding,
                'n_channels': n_channels,
            }

        # Create RGB grid (start with black border color: R=0, G=0, B=0)
        grid = self.np.zeros((grid_h, grid_w, 3), dtype=self.np.float32)
        # All channels stay 0 for black

        for i in range(n_channels):
            row = i // grid_cols
            col = i % grid_cols

            # Add padding offset to position each activation map
            y_start = row * (H + padding) + padding
            x_start = col * (W + padding) + padding

            # Normalize activation to 0-255 range for each map individually
            act = act_np[i]
            a_min, a_max = act.min(), act.max()
            if a_max > a_min:
                act_norm = (act - a_min) / (a_max - a_min) * 255
            else:
                act_norm = self.np.zeros_like(act)

            # Place grayscale activation in all RGB channels (creates grayscale appearance)
            grid[y_start:y_start + H, x_start:x_start + W, 0] = act_norm  # R
            grid[y_start:y_start + H, x_start:x_start + W, 1] = act_norm  # G
            grid[y_start:y_start + H, x_start:x_start + W, 2] = act_norm  # B

        return grid

    def update(self):
        """Update activation visualizations (throttled)."""
        if not self.is_open:
            return

        # Only update if enough time has passed
        current_time = self.time.time()
        if current_time - self.last_update_time >= self.update_interval:
            if self.needs_update:
                self._update_activations()
                self.last_update_time = current_time
                self.needs_update = False

            # Process events for pyqtgraph
            if self.use_pyqtgraph:
                from pyqtgraph.Qt import QtWidgets
                QtWidgets.QApplication.processEvents()

    def _update_activations(self):
        """Render activation maps for all layers."""
        if not self.is_open:
            return

        try:
            for idx, info in enumerate(self.conv_info):
                activation = info.get('activation')
                if activation is None:
                    continue

                # Create activation grid (now returns RGB)
                act_grid = self._create_activation_grid(activation, store_layout_idx=idx)

                # Store activation grid dimensions for window sizing (H, W, 3)
                info['activation_grid_shape'] = act_grid.shape[:2]  # (H, W)

                # Display based on backend
                if self.use_pyqtgraph:
                    # Activation grid is already 0-255 RGB, just convert to uint8
                    act_grid_norm = act_grid.astype(self.np.uint8)
                    # PyQtGraph expects (W, H, 3) format for RGB images, but numpy creates (H, W, 3)
                    # Transpose spatial dimensions: (H, W, 3) -> (W, H, 3)
                    act_grid_transposed = self.np.transpose(act_grid_norm, (1, 0, 2))
                    self.activation_items[idx].setImage(act_grid_transposed)

                    # Set view range to match transposed dimensions
                    h, w = act_grid.shape[:2]
                    self.activation_views[idx].setRange(xRange=(0, h), yRange=(0, w), padding=0)
                else:
                    self.axes[idx, 1].clear()
                    # Display RGB image (no colormap)
                    act_grid_norm = act_grid.astype(self.np.uint8)
                    self.axes[idx, 1].imshow(act_grid_norm)
                    self.axes[idx, 1].set_title(f"Layer {idx} Activations", fontsize=9)
                    self.axes[idx, 1].axis('off')

            # Auto-size window on first update (after we have both filters and activations)
            if self.use_pyqtgraph and not self._window_sized:
                self._autosize_window()
                self._window_sized = True

            if not self.use_pyqtgraph:
                # Keep tight layout
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            # Update detail viewer with new activation data
            if self.detail_viewer is not None and self.detail_viewer.is_open:
                # Get activation for the currently displayed layer and filter
                info = self.conv_info[self.detail_viewer.layer_idx]
                activation = info.get('activation')
                if activation is not None:
                    # activation shape: (batch, channels, H, W) or (channels, H, W)
                    if activation.dim() == 4:
                        activation = activation[0]  # Take first batch
                    if self.detail_viewer.filter_idx < activation.shape[0]:
                        activation_2d = activation[self.detail_viewer.filter_idx].cpu().numpy()
                        self.detail_viewer.update_activation(activation_2d)

        except Exception as e:
            import sys
            print(f"Error updating CNN filter/activation viewer: {e}", file=sys.stderr)
            self.is_open = False

    def _autosize_window(self):
        """Automatically resize window to fit content exactly."""
        if not self.use_pyqtgraph:
            return

        # Calculate required dimensions
        max_row_height = 0
        total_height = 0
        max_width = 0

        for info in self.conv_info:
            filter_shape = info.get('filter_grid_shape')
            act_shape = info.get('activation_grid_shape')

            if filter_shape and act_shape:
                # Height is the max of filter and activation for this row
                row_height = max(filter_shape[0], act_shape[0])
                total_height += row_height

                # Width is sum of filter and activation
                row_width = filter_shape[1] + act_shape[1]
                max_width = max(max_width, row_width)

        if max_width > 0 and total_height > 0:
            # Add small buffer for title bar (30px) and borders
            window_width = int(max_width + 10)
            window_height = int(total_height + 40)

            # Resize window to exactly fit content
            self.win.resize(window_width, window_height)
            print(f"  Auto-sized window to {window_width}x{window_height} pixels")

    def close(self):
        """Close the viewer and remove hooks."""
        # Remove forward hooks
        for handle in self.activation_handles:
            handle.remove()
        self.activation_handles.clear()

        # Close detail viewer if open
        if self.detail_viewer is not None:
            self.detail_viewer.close()
            self.detail_viewer = None

        if self.is_open:
            # Flush any pending updates
            if self.needs_update:
                try:
                    self._update_activations()
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


class PreprocessedObservationViewer:
    """Real-time preprocessed observation viewer for visualizing what the agent sees."""

    def __init__(self, update_interval: float = 0.05):
        """Initialize the viewer for displaying preprocessed observations.

        Args:
            update_interval: Minimum time (in seconds) between updates to throttle rendering
        """
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtWidgets
            import time

            self.pg = pg
            self.time = time
            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            # Create window (don't show yet)
            self.win = pg.GraphicsLayoutWidget(show=False, title="Preprocessed Observations")
            self.win.resize(800, 600)
            self.win.setWindowTitle('Preprocessed Observations (Agent View)')

            # Set window flags before showing
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'observation_viewer' in layout:
                pos = layout['observation_viewer']
                self.win.move(pos['x'], pos['y'])
                if pos['width'] and pos['height']:
                    self.win.resize(pos['width'], pos['height'])
            else:
                # Default position: right of the reward plot
                self.win.move(1300, 50)

            # Create image display widget
            self.view = self.win.addViewBox()
            self.view.setAspectLocked(True)
            self.view.invertY(True)  # Invert Y axis so origin is at top-left (image convention)
            # Remove all padding/margins around the image
            self.view.setContentsMargins(0, 0, 0, 0)
            self.view.setMenuEnabled(False)  # Disable right-click menu and border
            self.view.setMouseEnabled(x=False, y=False)  # Disable panning
            self.view.enableAutoRange(enable=False)  # Disable auto-range
            self.win.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.win.ci.layout.setSpacing(0)
            self.img_item = pg.ImageItem()
            self.view.addItem(self.img_item)

            # Track if window is still open
            self.is_open = True
            self.use_pyqtgraph = True

            # Store last observation for rendering
            self.last_obs = None
            self._window_sized = False  # Track if we've sized the window to match image

            # Install keyboard shortcuts (Ctrl+S to save layout)
            install_keyboard_shortcuts(self.win)

            # Show window after all setup is complete
            self.win.show()
            self.win.raise_()

        except ImportError:
            # Fallback to matplotlib if pyqtgraph is not available
            import matplotlib.pyplot as plt
            import time

            self.plt = plt
            self.time = time
            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            self.plt.ion()
            # Create figure with minimal padding
            self.fig, self.ax = self.plt.subplots(1, 1, figsize=(8, 6))
            self.fig.suptitle('Preprocessed Observations (Agent View)', fontsize=14, fontweight='bold')

            # Position window to the right of the reward plot
            try:
                manager = self.plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    # Try to set position (works with TkAgg backend)
                    manager.window.wm_geometry("+1300+50")
            except Exception:
                pass  # If positioning fails, continue anyway

            # Remove all whitespace around the image
            self.fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0, hspace=0)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_frame_on(False)  # Remove the axes frame

            self.is_open = True
            self.fig.canvas.mpl_connect('close_event', self._on_close)

            self.plt.tight_layout()
            self.plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.use_pyqtgraph = False
            self.last_obs = None
            self._window_sized = False  # Track if we've sized the window to match image

    def _on_close(self, event):
        """Handle window close event."""
        self.is_open = False

    def update(self, obs):
        """Update the viewer with a new observation (throttled).

        Args:
            obs: Observation array from environment (single env, shape varies)
        """
        if not self.is_open:
            return

        import numpy as np

        # Skip if obs is None or empty
        if obs is None or (isinstance(obs, np.ndarray) and obs.size == 0):
            return

        # Store observation
        self.last_obs = np.array(obs)
        self.needs_update = True

        # Only update display if enough time has passed (throttling)
        current_time = self.time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._update_display()
            self.last_update_time = current_time
            self.needs_update = False

            # Process events for pyqtgraph
            if self.use_pyqtgraph:
                from pyqtgraph.Qt import QtWidgets
                QtWidgets.QApplication.processEvents()

    def _update_display(self):
        """Update the display with the current observation."""
        if not self.is_open or self.last_obs is None:
            return

        import numpy as np

        try:
            obs = self.last_obs

            # Determine observation shape and type
            # Expected shapes:
            # - Vector: (D,)
            # - Single frame grayscale: (H, W) or (1, H, W)
            # - Single frame RGB: (3, H, W)
            # - Framestack grayscale: (N, H, W) where N is stack size
            # - Framestack RGB: (N*3, H, W) where N is stack size

            # Skip vector observations (already handled by VecObsBarPrinter)
            if obs.ndim == 1:
                return

            # Handle image observations
            if obs.ndim == 2:
                # Single grayscale frame (H, W)
                display_img = obs
            elif obs.ndim == 3:
                # Could be (C, H, W) format
                C, H, W = obs.shape

                if C == 1:
                    # Single grayscale frame with channel dim
                    display_img = obs[0]
                elif C == 3:
                    # Single RGB frame - convert to HWC for display
                    display_img = np.transpose(obs, (1, 2, 0))
                else:
                    # Framestack: C is the number of stacked frames
                    # Display frames vertically with frame 0 (oldest) at top
                    n_frames = C
                    frames = []
                    for i in range(n_frames):
                        frames.append(obs[i])

                    # Vertical layout: stack frames top to bottom
                    # Frame 0 (oldest) at top, Frame N-1 (newest) at bottom
                    display_img = np.concatenate(frames, axis=0)
            else:
                # Unsupported shape
                return

            # Normalize to 0-255 range for display
            if display_img.dtype == np.float32 or display_img.dtype == np.float64:
                # Assume normalized to [0, 1] or [-1, 1]
                img_min, img_max = display_img.min(), display_img.max()
                if img_min < 0:
                    # Assume [-1, 1] range
                    display_img = ((display_img + 1) * 127.5).astype(np.uint8)
                else:
                    # Assume [0, 1] range
                    display_img = (display_img * 255).astype(np.uint8)
            else:
                # Already uint8
                display_img = display_img.astype(np.uint8)

            # Update display
            if self.use_pyqtgraph:
                # PyQtGraph expects transposed image
                self.img_item.setImage(display_img.T, levels=(0, 255))

                # Resize window and set view range to match image size on first display
                if not self._window_sized:
                    # Get image dimensions (after transpose for PyQtGraph)
                    img_height, img_width = display_img.shape[:2]
                    # Set view range to exactly match the image dimensions (no padding)
                    # PyQtGraph uses transposed coordinates
                    self.view.setRange(xRange=(0, img_width), yRange=(0, img_height), padding=0)
                    # Scale up small images for better visibility (2x or 3x max)
                    scale = 1
                    if img_width < 150 or img_height < 150:
                        # Use smaller scale factor
                        scale = 2 if max(img_width, img_height) < 100 else 1
                    # Window size should exactly match the scaled image to avoid padding
                    # Account for title bar and borders
                    window_width = img_width * scale
                    window_height = img_height * scale + 20  # Minimal adjustment for title bar
                    self.win.resize(int(window_width), int(window_height))
                    self._window_sized = True
            else:
                # Matplotlib
                self.ax.clear()
                self.ax.imshow(display_img, cmap='gray' if display_img.ndim == 2 else None, vmin=0, vmax=255, aspect='auto')
                self.ax.set_title(f'Shape: {obs.shape}')
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_frame_on(False)  # Remove the axes frame
                self.ax.axis('off')  # Turn off axis completely

                # Resize figure to match image aspect ratio on first display
                if not self._window_sized:
                    img_height, img_width = display_img.shape[:2]
                    # Calculate figure size to match image aspect ratio
                    # Use a base width and calculate height to maintain aspect ratio
                    base_width = 8
                    aspect_ratio = img_height / img_width if img_width > 0 else 1
                    fig_height = base_width * aspect_ratio
                    self.fig.set_size_inches(base_width, fig_height)
                    self._window_sized = True

                self.plt.tight_layout(pad=0)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        except Exception as e:
            # Log error and mark viewer as closed to prevent further crashes
            import sys
            print(f"Error updating preprocessed observation viewer: {e}", file=sys.stderr)
            self.is_open = False

    def close(self):
        """Close the viewer window, flushing any pending updates first."""
        if self.is_open:
            # Flush any pending updates before closing
            if self.needs_update:
                try:
                    self._update_display()
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


def play_episodes_manual(env, target_episodes: int, mode: str, step_by_step: bool = False, fps: int | None = None, action_labels: dict[int, str] | None = None, plotter: RewardPlotter | None = None, obs_viewer: PreprocessedObservationViewer | None = None):
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

    try:
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

        # Update observation viewer
        if obs_viewer and obs_viewer.is_open:
            obs_viewer.update(obs[0])

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
    except KeyboardInterrupt:
        print("\nInterrupted by user")

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
    p.add_argument("--plot-metrics", dest="plot_metrics", action="store_true", default=True, help="Show real-time reward plot (default: True)")
    p.add_argument("--no-plot-metrics", dest="plot_metrics", action="store_false", help="Disable real-time reward plot")
    p.add_argument("--plot-update-interval", type=float, default=0.2, help="Minimum time (seconds) between plot updates (default: 0.2, lower=smoother but slower game)")
    p.add_argument("--plot-window-size", type=int, default=100, help="Number of steps to show in sliding window (default: 100)")
    p.add_argument("--show-preprocessing", dest="show_preprocessing", action="store_true", default=False, help="Show preprocessed observations in separate window (what agent sees)")
    p.add_argument("--show-cnn-filters", dest="show_cnn_filters", action="store_true", default=False, help="Show CNN filters and activations in separate window (requires CNN policy)")
    p.add_argument(
        "--env-kwargs",
        action="append",
        dest="env_kwargs",
        metavar="KEY=VALUE",
        help="Override env_kwargs fields (e.g., --env-kwargs state=Level2-1). Can be specified multiple times.",
    )
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

        # Apply env_kwargs overrides
        if args.env_kwargs:
            from utils.train_launcher import _apply_env_kwargs_overrides
            config = _apply_env_kwargs_overrides(config, args.env_kwargs)

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

        # Apply env_kwargs overrides
        if args.env_kwargs:
            from utils.train_launcher import _apply_env_kwargs_overrides
            config = _apply_env_kwargs_overrides(config, args.env_kwargs)

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
    if args.plot_metrics and not args.headless:
        try:
            plotter = RewardPlotter(
                update_interval=args.plot_update_interval,
                max_steps_shown=args.plot_window_size
            )
            _active_viewers['reward_plotter'] = plotter
            backend = "PyQtGraph (fast)" if plotter.use_pyqtgraph else "Matplotlib"
            print(f"Real-time reward plot enabled using {backend}")
            print(f"  Update interval: {args.plot_update_interval}s | Window size: {args.plot_window_size} steps")
            print("Close the plot window to disable plotting.")
        except Exception as e:
            print(f"Warning: Could not initialize reward plotter: {e}")
            plotter = None

    # Initialize preprocessed observation viewer if enabled and not headless
    obs_viewer = None
    if args.show_preprocessing and not args.headless:
        try:
            obs_viewer = PreprocessedObservationViewer(update_interval=0.05)
            _active_viewers['observation_viewer'] = obs_viewer
            backend = "PyQtGraph (fast)" if obs_viewer.use_pyqtgraph else "Matplotlib"
            print(f"Preprocessed observation viewer enabled using {backend}")
            print("Close the viewer window to disable it.")
        except Exception as e:
            print(f"Warning: Could not initialize preprocessed observation viewer: {e}")
            obs_viewer = None

    # Handle different modes
    if args.mode in ["random", "user"]:
        # Manual control modes don't need policy
        try:
            play_episodes_manual(env, target_episodes, args.mode, args.step_by_step, args.fps, action_labels, plotter, obs_viewer)
        finally:
            if plotter:
                plotter.close()
            if obs_viewer:
                obs_viewer.close()
        print("Done.")
        return

    # Trained mode: load policy
    assert run is not None, "run must be loaded for trained mode"
    # TODO: we should be loading the agent and having it run the episode
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

    # Initialize CNN filter/activation viewer if enabled and not headless
    cnn_viewer = None
    if args.show_cnn_filters and not args.headless:
        try:
            cnn_viewer = CNNFilterActivationViewer(policy_model, update_interval=0.1)
            _active_viewers['cnn_filter_viewer'] = cnn_viewer
            backend = "PyQtGraph (fast)" if cnn_viewer.use_pyqtgraph else "Matplotlib"
            print(f"CNN filter/activation viewer enabled using {backend}")
            print("Close the viewer window to disable it.")
        except ValueError as e:
            print(f"Warning: {e}")
            cnn_viewer = None
        except Exception as e:
            print(f"Warning: Could not initialize CNN filter/activation viewer: {e}")
            cnn_viewer = None

    # Initialize rollout collector; step-by-step mode, FPS limiting, plotting, or obs viewer uses single-step rollouts
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1 if (args.step_by_step or args.fps or plotter or obs_viewer or cnn_viewer) else config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    # Print hotkey instructions if any viewers are active
    if plotter or obs_viewer or cnn_viewer:
        print("\n" + "="*60)
        print("💡 TIP: Press F9 in any visualization window to save")
        print("   the current window layout to window_layout.json")
        print("="*60 + "\n")

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
            rollout = collector.collect(deterministic=args.deterministic)

            # Update plotter with step reward if plotting is enabled
            if plotter and plotter.is_open and collector.n_steps == 1:
                # When n_steps=1, buffer has exactly one step at the most recent position
                # Access the buffer's rewards_buf directly (shape: [n_steps, n_envs])
                step_reward = collector._buffer.rewards_buf[collector._buffer.pos - 1, 0]
                plotter.add_step(float(step_reward))

            # Update observation viewer if enabled
            if obs_viewer and obs_viewer.is_open and collector.n_steps == 1:
                # Access the most recent observation from the buffer
                obs = collector._buffer.obs_buf[collector._buffer.pos - 1, 0]
                obs_viewer.update(obs)

            # Update CNN filter/activation viewer if enabled
            if cnn_viewer and cnn_viewer.is_open:
                cnn_viewer.update()

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
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if plotter:
            plotter.close()
        if obs_viewer:
            obs_viewer.close()
        if cnn_viewer:
            cnn_viewer.close()

    print("Done.")


if __name__ == "__main__":
    main()
