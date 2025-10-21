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
    'action_visualizer': None,
    'cnn_filter_viewer': None,
    'cnn_detail_viewer': None,
    'maximal_activation_viewer': None,
    'rf_overlay_viewer': None
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


def install_keyboard_shortcuts(win, additional_handlers=None):
    """Install keyboard shortcuts on a PyQtGraph window.

    Args:
        win: PyQtGraph window
        additional_handlers: Dict of {key_code: handler_function} for additional shortcuts
    """
    try:
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        class KeyPressFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    # Check for F9 key (save layout)
                    if event.key() == QtCore.Qt.Key.Key_F9:
                        try:
                            # Use QTimer to defer save to avoid blocking the event loop
                            QtCore.QTimer.singleShot(0, save_window_layout)
                        except Exception as e:
                            print(f"Error in hotkey handler: {e}")
                        return False  # Don't consume the event to prevent freezing

                    # Check for additional handlers
                    if additional_handlers:
                        for key_code, handler in additional_handlers.items():
                            if event.key() == key_code:
                                try:
                                    QtCore.QTimer.singleShot(0, handler)
                                except Exception as e:
                                    print(f"Error in additional hotkey handler: {e}")
                                return False

                return False  # Don't consume other events

        # Create and install event filter
        key_filter = KeyPressFilter(win)
        win.installEventFilter(key_filter)
        # Store reference to prevent garbage collection
        win._key_filter = key_filter

    except Exception as e:
        print(f"Warning: Could not install keyboard shortcuts: {e}")


class VisualizationToolbar:
    """Interactive toolbar for toggling visualizations and changing color palettes."""

    def __init__(self, config, policy_model=None, env=None, reward_plotter=None, obs_viewer=None, action_viewer=None, cnn_viewer=None):
        """Initialize the visualization toolbar.

        Args:
            config: Configuration object with environment specs
            policy_model: Policy model (for CNN viewer)
            env: Environment instance
            reward_plotter: Existing reward plotter instance (if already created)
            obs_viewer: Existing observation viewer instance (if already created)
            action_viewer: Existing action visualizer instance (if already created)
            cnn_viewer: Existing CNN viewer instance (if already created)
        """
        try:
            import pygame
            self.pygame = pygame

            pygame.init()

            # Toolbar dimensions (more vertical, less horizontal)
            self.width = 280
            self.height = 435
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Visualization Toolbar")

            # Colors
            self.bg_color = (30, 30, 35)
            self.text_color = (220, 220, 220)
            self.accent_color = (80, 160, 255)
            self.active_color = (80, 200, 120)
            self.inactive_color = (120, 120, 130)
            self.button_bg = (50, 50, 55)
            self.button_hover = (70, 70, 75)
            self.button_active = (60, 140, 90)
            self.button_border = (100, 100, 110)

            # Fonts
            self.title_font = pygame.font.Font(None, 28)
            self.normal_font = pygame.font.Font(None, 22)
            self.small_font = pygame.font.Font(None, 18)
            self.icon_font = pygame.font.Font(None, 32)

            # Button definitions (x, y, width, height, action, icon, label)
            self.buttons = []
            self.hovered_button = None
            self.mouse_pos = (0, 0)

            # Store references
            self.config = config
            self.policy_model = policy_model
            self.env = env

            # Active viewer instances (accept existing ones)
            self.reward_plotter = reward_plotter
            self.obs_viewer = obs_viewer
            self.action_viewer = action_viewer
            self.cnn_viewer = cnn_viewer

            # Visualization states (initialize based on existing viewers)
            self.states = {
                'reward': reward_plotter is not None,
                'observation': obs_viewer is not None,
                'actions': action_viewer is not None,
                'filters': cnn_viewer is not None
            }

            # Color palettes for CNN filters/activations
            self.filter_palettes = ['RdBu', 'RdYlBu', 'Seismic', 'Coolwarm']
            self.activation_palettes = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
            self.current_filter_palette = 0
            self.current_activation_palette = 0

            # Window is open
            self.is_open = True

            # Initial render
            self._render()

        except ImportError:
            raise ImportError("pygame is required for visualization toolbar")

    def _render(self):
        """Render the toolbar UI."""
        self.screen.fill(self.bg_color)

        # Clear buttons list for redrawing
        self.buttons = []
        y_offset = 20

        # Section: Viewers
        section_title = self.small_font.render("Viewers:", True, self.accent_color)
        self.screen.blit(section_title, (20, y_offset))
        y_offset += 30

        # Rewards button
        btn_rect = self._draw_toggle_button(
            x=20, y=y_offset,
            label="Rewards",
            is_active=self.states['reward'],
            action='toggle_reward'
        )
        self.buttons.append(btn_rect)
        y_offset += 55

        # Observations button
        btn_rect = self._draw_toggle_button(
            x=20, y=y_offset,
            label="Observations",
            is_active=self.states['observation'],
            action='toggle_observation'
        )
        self.buttons.append(btn_rect)
        y_offset += 55

        # Actions button
        btn_rect = self._draw_toggle_button(
            x=20, y=y_offset,
            label="Actions",
            is_active=self.states['actions'],
            action='toggle_actions'
        )
        self.buttons.append(btn_rect)
        y_offset += 55

        # Filters button
        btn_rect = self._draw_toggle_button(
            x=20, y=y_offset,
            label="Filters",
            is_active=self.states['filters'],
            action='toggle_filters'
        )
        self.buttons.append(btn_rect)
        y_offset += 65

        # Section: Colormaps (only show if filters active)
        if self.states['filters']:
            section_title = self.small_font.render("Colormaps:", True, self.accent_color)
            self.screen.blit(section_title, (20, y_offset))
            y_offset += 30

            # Filter colormap selector
            filter_cmap = self.filter_palettes[self.current_filter_palette]
            btn_left = self._draw_arrow_button(
                x=20, y=y_offset,
                direction="<",
                action='cycle_filter_palette_prev'
            )
            self.buttons.append(btn_left)

            label_text = self.small_font.render(f"Filter: {filter_cmap}", True, self.text_color)
            self.screen.blit(label_text, (70, y_offset + 8))

            btn_right = self._draw_arrow_button(
                x=220, y=y_offset,
                direction=">",
                action='cycle_filter_palette_next'
            )
            self.buttons.append(btn_right)
            y_offset += 40

            # Activation colormap selector
            act_cmap = self.activation_palettes[self.current_activation_palette]
            btn_left = self._draw_arrow_button(
                x=20, y=y_offset,
                direction="<",
                action='cycle_activation_palette_prev'
            )
            self.buttons.append(btn_left)

            label_text = self.small_font.render(f"Activation: {act_cmap}", True, self.text_color)
            self.screen.blit(label_text, (70, y_offset + 8))

            btn_right = self._draw_arrow_button(
                x=220, y=y_offset,
                direction=">",
                action='cycle_activation_palette_next'
            )
            self.buttons.append(btn_right)

        self.pygame.display.flip()

    def _draw_toggle_button(self, x, y, label, is_active, action):
        """Draw a toggle button and return its rect."""
        width = 240
        height = 45

        # Check if mouse is hovering
        mouse_x, mouse_y = self.mouse_pos
        is_hovered = x <= mouse_x <= x + width and y <= mouse_y <= y + height

        # Determine button color
        if is_active:
            bg_color = self.button_active
            border_color = self.active_color
        elif is_hovered:
            bg_color = self.button_hover
            border_color = self.button_border
        else:
            bg_color = self.button_bg
            border_color = self.button_border

        # Draw button background
        self.pygame.draw.rect(self.screen, bg_color, (x, y, width, height), border_radius=6)
        # Draw border
        self.pygame.draw.rect(self.screen, border_color, (x, y, width, height), width=2, border_radius=6)

        # Draw label
        label_surf = self.normal_font.render(label, True, self.text_color)
        self.screen.blit(label_surf, (x + 15, y + 12))

        # Draw status indicator
        status_text = "ON" if is_active else "OFF"
        status_color = self.active_color if is_active else self.inactive_color
        status_surf = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status_surf, (x + 180, y + 14))

        return {'rect': self.pygame.Rect(x, y, width, height), 'action': action}

    def _draw_arrow_button(self, x, y, direction, action):
        """Draw an arrow button for cycling options."""
        width = 40
        height = 30

        # Check if mouse is hovering
        mouse_x, mouse_y = self.mouse_pos
        is_hovered = x <= mouse_x <= x + width and y <= mouse_y <= y + height

        # Determine button color
        bg_color = self.button_hover if is_hovered else self.button_bg
        border_color = self.accent_color if is_hovered else self.button_border

        # Draw button
        self.pygame.draw.rect(self.screen, bg_color, (x, y, width, height), border_radius=4)
        self.pygame.draw.rect(self.screen, border_color, (x, y, width, height), width=2, border_radius=4)

        # Draw arrow
        arrow_surf = self.normal_font.render(direction, True, self.text_color)
        arrow_rect = arrow_surf.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(arrow_surf, arrow_rect)

        return {'rect': self.pygame.Rect(x, y, width, height), 'action': action}

    def handle_events(self):
        """Process toolbar events and return active viewer instances."""
        if not self.is_open:
            return None, None, None, None

        # Update mouse position
        self.mouse_pos = self.pygame.mouse.get_pos()

        # Process all pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                return None, None, None, None
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_q:
                    self.close()
                    return None, None, None, None
                elif event.key == self.pygame.K_r:
                    self._toggle_reward()
                elif event.key == self.pygame.K_o:
                    self._toggle_observation()
                elif event.key == self.pygame.K_a:
                    self._toggle_actions()
                elif event.key == self.pygame.K_f:
                    self._toggle_filters()
                elif event.key == self.pygame.K_LEFTBRACKET:
                    self._cycle_filter_palette_prev()
                elif event.key == self.pygame.K_RIGHTBRACKET:
                    self._cycle_activation_palette_next()
            elif event.type == self.pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_button_click(event.pos)

        # Re-render on every call to keep window responsive
        self._render()

        return self.reward_plotter, self.obs_viewer, self.action_viewer, self.cnn_viewer

    def _handle_button_click(self, pos):
        """Handle mouse clicks on buttons."""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                action = button['action']

                # Execute the appropriate action
                if action == 'toggle_reward':
                    self._toggle_reward()
                elif action == 'toggle_observation':
                    self._toggle_observation()
                elif action == 'toggle_actions':
                    self._toggle_actions()
                elif action == 'toggle_filters':
                    self._toggle_filters()
                elif action == 'cycle_filter_palette_prev':
                    self._cycle_filter_palette_prev()
                elif action == 'cycle_filter_palette_next':
                    self._cycle_filter_palette_next()
                elif action == 'cycle_activation_palette_prev':
                    self._cycle_activation_palette_prev()
                elif action == 'cycle_activation_palette_next':
                    self._cycle_activation_palette_next()
                break

    def _toggle_reward(self):
        """Toggle reward plotter."""
        self.states['reward'] = not self.states['reward']

        if self.states['reward']:
            # Create reward plotter
            try:
                self.reward_plotter = RewardPlotter(update_interval=0.2, max_steps_shown=100)
                _active_viewers['reward_plotter'] = self.reward_plotter
                print("Reward plotter enabled")
                # Restore focus to toolbar
                self._restore_focus()
            except Exception as e:
                print(f"Failed to create reward plotter: {e}")
                self.states['reward'] = False
        else:
            # Close reward plotter
            if self.reward_plotter:
                self.reward_plotter.close()
                self.reward_plotter = None
                _active_viewers['reward_plotter'] = None
                print("Reward plotter disabled")

    def _toggle_observation(self):
        """Toggle observation viewer."""
        self.states['observation'] = not self.states['observation']

        if self.states['observation']:
            # Create observation viewer
            try:
                self.obs_viewer = PreprocessedObservationViewer(update_interval=0.05)
                _active_viewers['observation_viewer'] = self.obs_viewer
                print("Observation viewer enabled")
                # Restore focus to toolbar
                self._restore_focus()
            except Exception as e:
                print(f"Failed to create observation viewer: {e}")
                self.states['observation'] = False
        else:
            # Close observation viewer
            if self.obs_viewer:
                self.obs_viewer.close()
                self.obs_viewer = None
                _active_viewers['observation_viewer'] = None
                print("Observation viewer disabled")

    def _toggle_actions(self):
        """Toggle action visualizer."""
        self.states['actions'] = not self.states['actions']

        if self.states['actions']:
            # Create action visualizer
            if self.env is None:
                print("Cannot enable action visualizer: no environment loaded")
                self.states['actions'] = False
                self._render()  # Re-render to show correct state
                return

            try:
                # Get number of actions from environment
                from gymnasium.spaces import Discrete, MultiBinary
                action_space = self.env.single_action_space
                if not isinstance(action_space, (Discrete, MultiBinary)):
                    print(f"Action visualizer only supports Discrete and MultiBinary action spaces (got {type(action_space).__name__})")
                    self.states['actions'] = False
                    self._render()  # Re-render to show correct state
                    return

                n_actions = action_space.n

                # Extract action labels from config
                action_labels = extract_action_labels_from_config(self.config)
                if action_labels:
                    print(f"Loaded {len(action_labels)} action labels from config spec")
                else:
                    print("No action labels found in config spec")

                space_type = "MultiBinary buttons" if isinstance(action_space, MultiBinary) else "Discrete actions"
                self.action_viewer = ActionVisualizer(n_actions=n_actions, action_labels=action_labels, update_interval=0.05)
                _active_viewers['action_visualizer'] = self.action_viewer
                print("Action visualizer enabled")
                # Restore focus to toolbar
                self._restore_focus()
            except Exception as e:
                print(f"Failed to create action visualizer: {e}")
                import traceback
                traceback.print_exc()
                self.states['actions'] = False
                self._render()  # Re-render to show correct state
        else:
            # Close action visualizer
            if self.action_viewer:
                self.action_viewer.close()
                self.action_viewer = None
                _active_viewers['action_visualizer'] = None
                print("Action visualizer disabled")

    def _toggle_filters(self):
        """Toggle CNN filter viewer."""
        self.states['filters'] = not self.states['filters']

        if self.states['filters']:
            # Create CNN filter viewer
            if self.policy_model is None:
                print("Cannot enable filter viewer: no policy model loaded")
                self.states['filters'] = False
                return

            try:
                self.cnn_viewer = CNNFilterActivationViewer(self.policy_model, update_interval=0.1)
                _active_viewers['cnn_filter_viewer'] = self.cnn_viewer
                print("CNN filter viewer enabled")
                # Restore focus to toolbar
                self._restore_focus()
            except Exception as e:
                print(f"Failed to create CNN filter viewer: {e}")
                self.states['filters'] = False
                self.cnn_viewer = None
        else:
            # Close CNN filter viewer
            if self.cnn_viewer is not None:
                try:
                    self.cnn_viewer.close()
                except Exception as e:
                    print(f"Warning: Error closing CNN filter viewer: {e}")
                finally:
                    self.cnn_viewer = None
                    _active_viewers['cnn_filter_viewer'] = None
                    print("CNN filter viewer disabled")

    def _cycle_filter_palette_prev(self):
        """Cycle to previous filter color palette."""
        self.current_filter_palette = (self.current_filter_palette - 1) % len(self.filter_palettes)
        palette_name = self.filter_palettes[self.current_filter_palette]
        print(f"Filter colormap: {palette_name}")

        # Apply to active CNN viewer
        if self.cnn_viewer is not None and self.cnn_viewer.is_open:
            try:
                self.cnn_viewer.set_filter_colormap(palette_name)
                print(f"  Applied {palette_name} to filter viewer")
            except Exception as e:
                print(f"  Error applying colormap: {e}")

    def _cycle_filter_palette_next(self):
        """Cycle to next filter color palette."""
        self.current_filter_palette = (self.current_filter_palette + 1) % len(self.filter_palettes)
        palette_name = self.filter_palettes[self.current_filter_palette]
        print(f"Filter colormap: {palette_name}")

        # Apply to active CNN viewer
        if self.cnn_viewer is not None and self.cnn_viewer.is_open:
            try:
                self.cnn_viewer.set_filter_colormap(palette_name)
                print(f"  Applied {palette_name} to filter viewer")
            except Exception as e:
                print(f"  Error applying colormap: {e}")

    def _cycle_activation_palette_prev(self):
        """Cycle to previous activation color palette."""
        self.current_activation_palette = (self.current_activation_palette - 1) % len(self.activation_palettes)
        palette_name = self.activation_palettes[self.current_activation_palette]
        print(f"Activation colormap: {palette_name}")

        # Apply to active CNN viewer
        if self.cnn_viewer is not None and self.cnn_viewer.is_open:
            try:
                self.cnn_viewer.set_activation_colormap(palette_name)
                print(f"  Applied {palette_name} to activation viewer")
            except Exception as e:
                print(f"  Error applying colormap: {e}")

    def _cycle_activation_palette_next(self):
        """Cycle to next activation color palette."""
        self.current_activation_palette = (self.current_activation_palette + 1) % len(self.activation_palettes)
        palette_name = self.activation_palettes[self.current_activation_palette]
        print(f"Activation colormap: {palette_name}")

        # Apply to active CNN viewer
        if self.cnn_viewer is not None and self.cnn_viewer.is_open:
            try:
                self.cnn_viewer.set_activation_colormap(palette_name)
                print(f"  Applied {palette_name} to activation viewer")
            except Exception as e:
                print(f"  Error applying colormap: {e}")

    def _restore_focus(self):
        """Restore focus to the toolbar window after creating viewers."""
        try:
            # Give the new window a moment to finish initializing
            import time
            time.sleep(0.05)

            # Re-raise the toolbar window
            import os
            if os.name == 'posix':
                # On macOS/Linux, use SDL window management
                from pygame import display
                display.get_wm_info()  # This can help trigger focus

            # Force pygame window to front
            self.pygame.event.pump()
            self.pygame.display.flip()
        except Exception as e:
            # Focus restoration is best-effort, don't fail if it doesn't work
            pass

    def close(self):
        """Close the toolbar and all active viewers."""
        if not self.is_open:
            return

        self.is_open = False

        # Close all active viewers
        if self.reward_plotter:
            try:
                self.reward_plotter.close()
            except Exception:
                pass
        if self.obs_viewer:
            try:
                self.obs_viewer.close()
            except Exception:
                pass
        if self.cnn_viewer:
            try:
                self.cnn_viewer.close()
            except Exception:
                pass

        # Quit pygame
        try:
            self.pygame.quit()
        except Exception:
            pass


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


class ActionVisualizer:
    """Real-time action probability visualizer showing which actions are selected and their probabilities."""

    def __init__(self, n_actions: int, action_labels: dict[int, str] | None = None, update_interval: float = 0.05):
        """Initialize the action visualizer.

        Args:
            n_actions: Number of possible actions
            action_labels: Optional dict mapping action index to label string
            update_interval: Minimum time (in seconds) between plot updates to throttle rendering
        """
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore
            import time

            self.pg = pg
            self.time = time
            self.n_actions = n_actions
            self.action_labels = action_labels or {}

            # Set background to white for better visibility
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

            # Create compact window (don't show yet)
            self.win = pg.GraphicsLayoutWidget(show=False, title="Actions")
            # Compact size: width=250 to fit label + bar + value, height based on number of actions
            self.win.resize(250, max(60, 40 + n_actions * 22))
            self.win.setWindowTitle('Actions')

            # Set window flags before showing
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'action_visualizer' in layout:
                pos = layout['action_visualizer']
                self.win.move(pos['x'], pos['y'])
                if pos['width'] and pos['height']:
                    self.win.resize(pos['width'], pos['height'])
            else:
                # Default position: below reward plotter
                self.win.move(750, 500)

            # Create compact plot
            self.plot = self.win.addPlot()
            self.plot.hideAxis('left')
            self.plot.hideAxis('bottom')
            self.plot.setXRange(0, 1, padding=0)
            self.plot.setYRange(0, n_actions, padding=0)
            self.plot.setMouseEnabled(x=False, y=False)

            # Layout positions
            self.label_x = 0.02  # Left edge for labels
            self.bar_start_x = 0.35  # Where bars start
            self.bar_max_width = 0.58  # Maximum bar width (ends at ~0.93)
            self.value_x = 0.95  # Right edge for values

            # Create horizontal bars and text for each action
            self.bar_items = []
            self.label_items = []
            self.value_items = []

            for i in range(n_actions):
                # Invert y-position so action 0 is at top
                y_pos = n_actions - i - 0.5

                # Create action label (left side, left-aligned)
                label = action_labels.get(i, f"{i}") if action_labels else f"{i}"
                # Truncate long labels to fit
                if len(label) > 12:
                    label = label[:11] + '…'
                label_text = pg.TextItem(text=label, anchor=(0, 0.5), color='k')
                label_text.setPos(self.label_x, y_pos)
                self.plot.addItem(label_text)
                self.label_items.append(label_text)

                # Create horizontal bar (starts at bar_start_x, extends rightward)
                bar = pg.BarGraphItem(x=[self.bar_start_x], y=[y_pos], height=0.6, width=[0], brush='steelblue')
                self.plot.addItem(bar)
                self.bar_items.append(bar)

                # Create probability value (right side, right-aligned)
                value_text = pg.TextItem(text='0.00', anchor=(1, 0.5), color='k')
                value_text.setPos(self.value_x, y_pos)
                self.plot.addItem(value_text)
                self.value_items.append(value_text)

            # Store current state
            self.current_probs = [0.0] * n_actions
            self.current_action = None

            # Track if window is still open
            self.is_open = True

            # Connect window close event
            self.win.closeEvent = lambda event: setattr(self, 'is_open', False)

            # Throttling: only update plot at fixed intervals
            self.update_interval = update_interval
            self.last_update_time = time.time()
            self.needs_update = False

            self.use_pyqtgraph = True

            # Install keyboard shortcuts
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
            self.n_actions = n_actions
            self.action_labels = action_labels or {}
            self.plt.ion()

            self.fig, self.ax = self.plt.subplots(figsize=(4, max(2, min(6, 0.8 + n_actions * 0.3))))
            self.fig.suptitle('Action Probabilities', fontsize=9, fontweight='bold')

            # Position window
            try:
                manager = self.plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    manager.window.wm_geometry("+750+500")
            except Exception:
                pass

            self.ax.set_xlabel('Action', fontsize=7)
            self.ax.set_ylabel('Probability', fontsize=7)
            self.ax.set_ylim(0, 1.0)
            self.ax.grid(True, alpha=0.3, axis='y')
            self.ax.tick_params(labelsize=6)

            # Set x-axis labels
            x_positions = list(range(n_actions))
            if action_labels:
                labels = [action_labels.get(i, f"A{i}") for i in range(n_actions)]
                self.ax.set_xticks(x_positions)
                self.ax.set_xticklabels(labels, rotation=45 if n_actions > 8 else 0, ha='right' if n_actions > 8 else 'center')
            else:
                self.ax.set_xticks(x_positions)
                self.ax.set_xticklabels([str(i) for i in range(n_actions)])

            # Create initial bars
            self.bars = self.ax.bar(x_positions, [0] * n_actions, color='steelblue')

            # Store current state
            self.current_probs = [0.0] * n_actions
            self.current_action = None

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

    def update(self, action_probs: list[float] | None, selected_action: int | None):
        """Update the action probability visualization.

        Args:
            action_probs: List of probabilities for each action (length should match n_actions)
            selected_action: Index of the selected action (highlighted)
        """
        if not self.is_open:
            return

        # Store current state
        if action_probs is not None:
            self.current_probs = list(action_probs)
        self.current_action = selected_action
        self.needs_update = True

        # Check if enough time has passed since last update
        current_time = self.time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        # Perform the update
        self._update_display()
        self.last_update_time = current_time
        self.needs_update = False

    def _update_display(self):
        """Update the display with current probabilities."""
        if not self.is_open:
            return

        try:
            if self.use_pyqtgraph:
                # Update each bar and value text
                for i, prob in enumerate(self.current_probs):
                    y_pos = self.n_actions - i - 0.5

                    # Set color based on whether this action/button is selected/pressed
                    if i == self.current_action:
                        # Discrete: specific action selected
                        brush = self.pg.mkBrush((80, 200, 120))  # Green for selected
                    elif self.current_action is None and prob > 0.5:
                        # MultiBinary: button is pressed (state = 1)
                        brush = self.pg.mkBrush((80, 200, 120))  # Green for pressed
                    else:
                        # Not selected/pressed
                        brush = self.pg.mkBrush((31, 119, 180))  # Blue for others

                    # Update bar width (bar starts at bar_start_x and extends based on probability)
                    bar_width = prob * self.bar_max_width
                    # Center the bar so its left edge starts at bar_start_x
                    bar_center_x = self.bar_start_x + bar_width / 2
                    self.bar_items[i].setOpts(x=[bar_center_x], y=[y_pos], height=0.6, width=[bar_width], brush=brush)

                    # Update probability value text
                    self.value_items[i].setText(f'{prob:.2f}')

                # Process events to keep UI responsive
                self.pg.QtWidgets.QApplication.processEvents()
            else:
                # Matplotlib fallback
                for i, prob in enumerate(self.current_probs):
                    self.bars[i].set_height(prob)
                    if i == self.current_action:
                        # Discrete: specific action selected
                        self.bars[i].set_color('mediumseagreen')
                    elif self.current_action is None and prob > 0.5:
                        # MultiBinary: button is pressed (state = 1)
                        self.bars[i].set_color('mediumseagreen')
                    else:
                        # Not selected/pressed
                        self.bars[i].set_color('steelblue')

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Warning: Error updating action visualizer: {e}")

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


def calculate_receptive_field_info(conv_layers):
    """Calculate receptive field size, stride, and offset for each conv layer.

    Args:
        conv_layers: List of nn.Conv2d layers

    Returns:
        List of dicts with 'rf_size', 'rf_stride', 'rf_offset' for each layer
    """
    rf_info = []

    # Start with identity receptive field
    current_rf_size = 1      # Receptive field size
    current_stride = 1       # Stride from output to input
    current_offset = 0.5     # Center offset

    for layer in conv_layers:
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding

        # Update receptive field size
        # RF_out = RF_in + (kernel_size - 1) * stride_in
        new_rf_size = current_rf_size + (kernel_size - 1) * current_stride

        # Update stride (how much we move in input space per output pixel)
        new_stride = current_stride * stride

        # Update offset (center of first receptive field)
        # offset_out = offset_in + ((kernel_size - 1) / 2 - padding) * stride_in
        new_offset = current_offset + ((kernel_size - 1) / 2 - padding) * current_stride

        rf_info.append({
            'rf_size': new_rf_size,
            'rf_stride': new_stride,
            'rf_offset': new_offset,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
        })

        current_rf_size = new_rf_size
        current_stride = new_stride
        current_offset = new_offset

    return rf_info


def get_receptive_field_box(layer_rf_info, act_y, act_x, input_shape):
    """Get input bounding box for a specific activation position.

    Args:
        layer_rf_info: Dict with 'rf_size', 'rf_stride', 'rf_offset' for this layer
        act_y: Activation Y coordinate (row)
        act_x: Activation X coordinate (col)
        input_shape: Tuple (H, W) of input image

    Returns:
        Dict with 'y_min', 'y_max', 'x_min', 'x_max' in input coordinates
    """
    rf_size = layer_rf_info['rf_size']
    rf_stride = layer_rf_info['rf_stride']
    rf_offset = layer_rf_info['rf_offset']

    # Center of receptive field in input space
    center_y = act_y * rf_stride + rf_offset
    center_x = act_x * rf_stride + rf_offset

    # Bounding box (half-width on each side)
    half_size = rf_size / 2
    y_min = max(0, int(center_y - half_size))
    y_max = min(input_shape[0], int(center_y + half_size))
    x_min = max(0, int(center_x - half_size))
    x_max = min(input_shape[1], int(center_x + half_size))

    return {
        'y_min': y_min,
        'y_max': y_max,
        'x_min': x_min,
        'x_max': x_max,
        'center_y': center_y,
        'center_x': center_x,
    }


class TopKActivationBuffer:
    """Buffer to store top-K image patches that maximally activate a filter."""

    def __init__(self, k=10):
        """Initialize buffer.

        Args:
            k: Number of top activations to keep
        """
        import heapq
        self.k = k
        self.heap = []  # Min-heap of (activation_value, timestamp, patch_data)
        self.counter = 0  # Monotonic counter for tie-breaking and ordering
        self.heapq = heapq

    def update(self, activation_value, input_patch):
        """Update buffer with new activation.

        Args:
            activation_value: Scalar activation value (max or mean of activation map)
            input_patch: Input image patch (numpy array, typically CHW or HW)
        """
        import numpy as np

        # Use counter for stable ordering (more recent = higher priority in ties)
        self.counter += 1

        # Store copy of patch to avoid reference issues
        patch_copy = input_patch.copy() if isinstance(input_patch, np.ndarray) else input_patch

        if len(self.heap) < self.k:
            # Heap not full, just add
            self.heapq.heappush(self.heap, (activation_value, self.counter, patch_copy))
        elif activation_value > self.heap[0][0]:
            # New activation is larger than smallest in heap, replace
            self.heapq.heapreplace(self.heap, (activation_value, self.counter, patch_copy))

    def get_top_k(self):
        """Get top-K patches sorted by activation value (highest first).

        Returns:
            List of (activation_value, patch_data) tuples, sorted descending
        """
        # Sort heap by activation value (descending), then by counter (descending)
        sorted_items = sorted(self.heap, key=lambda x: (-x[0], -x[1]))
        return [(val, patch) for val, _, patch in sorted_items]

    def clear(self):
        """Clear all stored activations."""
        self.heap = []
        self.counter = 0


class MaximalActivationViewer:
    """Viewer for displaying top-K image patches that maximally activate a filter."""

    def __init__(self, layer_idx, filter_idx, buffer, parent_viewer=None):
        """Initialize maximal activation viewer.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index within the layer
            buffer: TopKActivationBuffer instance
            parent_viewer: Parent CNNFilterActivationViewer for coordination
        """
        try:
            import pyqtgraph as pg
            import numpy as np

            self.pg = pg
            self.np = np
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx
            self.buffer = buffer
            self.parent_viewer = parent_viewer

            # Set background to black
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')

            # Create window
            title = f"Layer {layer_idx} Filter {filter_idx} - Top Activations"
            self.win = pg.GraphicsLayoutWidget(show=False, title=title)
            self.win.setWindowTitle(title)

            # Set window flags to stay on top
            try:
                from pyqtgraph.Qt import QtCore
                self.win.setWindowFlags(self.win.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            except Exception:
                pass

            # Remove spacing
            self.win.ci.layout.setSpacing(1)
            self.win.ci.layout.setContentsMargins(2, 2, 2, 2)

            # Create grid of image items for top-K patches
            self.patch_items = []
            self.patch_views = []
            self.value_labels = []

            # Get top-K patches
            top_k_patches = buffer.get_top_k()

            if not top_k_patches:
                # No patches yet, show placeholder
                label = self.win.addLabel("No activations recorded yet.\nPlay some episodes to collect data.",
                                         color='gray', size='12pt')
                self.win.resize(400, 100)
            else:
                # Arrange in grid (2 columns for better viewing)
                n_patches = len(top_k_patches)
                n_cols = 2
                n_rows = (n_patches + n_cols - 1) // n_cols

                for idx, (act_val, patch) in enumerate(top_k_patches):
                    row = idx // n_cols
                    col = idx % n_cols

                    # Add view box for patch
                    view = self.win.addViewBox(row=row*2, col=col)
                    view.setAspectLocked(True)
                    view.invertY(True)
                    view.setMenuEnabled(False)
                    view.setMouseEnabled(x=False, y=False)
                    view.enableAutoRange(enable=False)

                    # Create image item
                    img_item = pg.ImageItem()
                    view.addItem(img_item)

                    # Render patch
                    patch_rgb = self._render_patch(patch)
                    if patch_rgb is not None:
                        # Transpose for PyQtGraph: (H, W, C) -> (W, H, C)
                        if len(patch_rgb.shape) == 3:
                            patch_rgb = self.np.transpose(patch_rgb, (1, 0, 2))
                        elif len(patch_rgb.shape) == 2:
                            patch_rgb = self.np.transpose(patch_rgb, (1, 0))

                        img_item.setImage(patch_rgb)
                        h, w = patch.shape[-2:] if len(patch.shape) >= 2 else patch.shape
                        view.setRange(xRange=(0, h), yRange=(0, w), padding=0)

                    self.patch_items.append(img_item)
                    self.patch_views.append(view)

                    # Add label below patch showing activation value
                    label = self.win.addLabel(f"Act: {act_val:.3f}",
                                             row=row*2+1, col=col,
                                             color='cyan', size='10pt')
                    self.value_labels.append(label)

                # Auto-size window based on patch dimensions
                if top_k_patches:
                    sample_patch = top_k_patches[0][1]
                    h, w = sample_patch.shape[-2:] if len(sample_patch.shape) >= 2 else sample_patch.shape
                    # Scale up small patches for visibility
                    scale = max(1, 80 // max(h, w))
                    patch_display_size = max(h, w) * scale
                    window_width = n_cols * patch_display_size + 20
                    window_height = n_rows * (patch_display_size + 30) + 20
                    self.win.resize(int(window_width), int(window_height))

            self.is_open = True
            self.has_data = len(top_k_patches) > 0  # Track if we have data to display

            # Set up close event handler
            def on_close(event):
                self.is_open = False
                if _active_viewers.get('maximal_activation_viewer') is self:
                    _active_viewers['maximal_activation_viewer'] = None
                event.accept()

            self.win.closeEvent = on_close

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'maximal_activation_viewer' in layout:
                pos = layout['maximal_activation_viewer']
                self.win.move(pos['x'], pos['y'])
            else:
                # Default position: to the right of detail viewer
                self.win.move(1100, 100)

            # Show window
            self.win.show()
            self.win.raise_()

        except ImportError:
            raise ImportError("PyQtGraph is required for maximal activation viewer")

    def refresh(self, layer_idx=None, filter_idx=None):
        """Refresh display with current buffer data.

        Args:
            layer_idx: New layer index (if changing filter)
            filter_idx: New filter index (if changing filter)
        """
        if not self.is_open:
            return

        # Update filter selection if provided
        if layer_idx is not None and filter_idx is not None:
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx
            self.win.setWindowTitle(f"Layer {layer_idx} Filter {filter_idx} - Top Activations")

            # Get new buffer
            if self.parent_viewer:
                self.buffer = self.parent_viewer.get_buffer(layer_idx, filter_idx)

        # Get current top-K patches
        if self.buffer is None:
            return

        top_k_patches = self.buffer.get_top_k()

        # If no data yet and we're showing placeholder, just return
        if not top_k_patches and not self.has_data:
            return

        # If we're transitioning from no data to having data, or vice versa, rebuild window
        if (not top_k_patches and self.has_data) or (top_k_patches and not self.has_data):
            self._rebuild_display(top_k_patches)
            return

        # Update existing displays in place
        if top_k_patches and len(top_k_patches) == len(self.patch_items):
            # Same number of items, just update values and images
            for idx, (act_val, patch) in enumerate(top_k_patches):
                # Update image
                patch_rgb = self._render_patch(patch)
                if patch_rgb is not None:
                    if len(patch_rgb.shape) == 3:
                        patch_rgb = self.np.transpose(patch_rgb, (1, 0, 2))
                    elif len(patch_rgb.shape) == 2:
                        patch_rgb = self.np.transpose(patch_rgb, (1, 0))
                    self.patch_items[idx].setImage(patch_rgb)

                # Update label
                self.value_labels[idx].setText(f"Act: {act_val:.3f}")
        else:
            # Different number of items, need to rebuild
            self._rebuild_display(top_k_patches)

    def _rebuild_display(self, top_k_patches):
        """Rebuild the entire display (when structure changes)."""
        # Clear existing layout
        self.win.clear()
        self.patch_items = []
        self.patch_views = []
        self.value_labels = []

        if not top_k_patches:
            # No patches yet, show placeholder
            self.win.addLabel("No activations recorded yet.\nPlay some episodes to collect data.",
                             color='gray', size='12pt')
            self.win.resize(400, 100)
            self.has_data = False
        else:
            # Arrange in grid (2 columns for better viewing)
            n_patches = len(top_k_patches)
            n_cols = 2
            n_rows = (n_patches + n_cols - 1) // n_cols

            for idx, (act_val, patch) in enumerate(top_k_patches):
                row = idx // n_cols
                col = idx % n_cols

                # Add view box for patch
                view = self.win.addViewBox(row=row*2, col=col)
                view.setAspectLocked(True)
                view.invertY(True)
                view.setMenuEnabled(False)
                view.setMouseEnabled(x=False, y=False)
                view.enableAutoRange(enable=False)

                # Create image item
                img_item = self.pg.ImageItem()
                view.addItem(img_item)

                # Render patch
                patch_rgb = self._render_patch(patch)
                if patch_rgb is not None:
                    if len(patch_rgb.shape) == 3:
                        patch_rgb = self.np.transpose(patch_rgb, (1, 0, 2))
                    elif len(patch_rgb.shape) == 2:
                        patch_rgb = self.np.transpose(patch_rgb, (1, 0))
                    img_item.setImage(patch_rgb)
                    h, w = patch.shape[-2:] if len(patch.shape) >= 2 else patch.shape
                    view.setRange(xRange=(0, h), yRange=(0, w), padding=0)

                self.patch_items.append(img_item)
                self.patch_views.append(view)

                # Add label below patch showing activation value
                label = self.win.addLabel(f"Act: {act_val:.3f}",
                                         row=row*2+1, col=col,
                                         color='cyan', size='10pt')
                self.value_labels.append(label)

            # Auto-size window based on patch dimensions
            if top_k_patches:
                sample_patch = top_k_patches[0][1]
                h, w = sample_patch.shape[-2:] if len(sample_patch.shape) >= 2 else sample_patch.shape
                scale = max(1, 80 // max(h, w))
                patch_display_size = max(h, w) * scale
                window_width = n_cols * patch_display_size + 20
                window_height = n_rows * (patch_display_size + 30) + 20
                self.win.resize(int(window_width), int(window_height))

            self.has_data = True

    def _render_patch(self, patch):
        """Render a patch as RGB or grayscale for display.

        Args:
            patch: Numpy array, typically (C, H, W) or (H, W)

        Returns:
            RGB or grayscale array ready for display
        """
        import numpy as np

        if patch is None or patch.size == 0:
            return None

        # Normalize for display
        patch_min, patch_max = patch.min(), patch.max()
        if patch_max > patch_min:
            patch_norm = (patch - patch_min) / (patch_max - patch_min)
        else:
            patch_norm = np.zeros_like(patch)

        # Convert to uint8
        patch_uint8 = (patch_norm * 255).astype(np.uint8)

        # Handle different formats
        if len(patch.shape) == 2:
            # Grayscale (H, W)
            return patch_uint8
        elif len(patch.shape) == 3:
            if patch.shape[0] == 1:
                # Single channel (1, H, W) -> (H, W)
                return patch_uint8[0]
            elif patch.shape[0] == 3:
                # RGB (3, H, W) -> (H, W, 3)
                return np.transpose(patch_uint8, (1, 2, 0))
            else:
                # Multi-channel, average to grayscale
                return patch_uint8.mean(axis=0).astype(np.uint8)

        return patch_uint8

    def close(self):
        """Close the viewer."""
        if self.is_open:
            try:
                self.win.close()
            except Exception:
                pass
            self.is_open = False
            if _active_viewers.get('maximal_activation_viewer') is self:
                _active_viewers['maximal_activation_viewer'] = None


class ReceptiveFieldOverlay:
    """Overlay showing receptive field box on input observation."""

    def __init__(self, layer_idx, filter_idx, rf_info, parent_viewer=None):
        """Initialize receptive field overlay viewer.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index
            rf_info: Receptive field info dict for this layer
            parent_viewer: Parent CNNFilterActivationViewer
        """
        try:
            import pyqtgraph as pg
            import numpy as np

            self.pg = pg
            self.np = np
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx
            self.rf_info = rf_info
            self.parent_viewer = parent_viewer

            # Set background to black
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')

            # Create window
            title = f"Layer {layer_idx} Filter {filter_idx} - Receptive Field"
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

            # Create view for input observation with RF overlay
            self.view = self.win.addViewBox()
            self.view.setAspectLocked(True)
            self.view.invertY(True)
            self.view.setMenuEnabled(False)
            self.view.setMouseEnabled(x=False, y=False)

            # Add image item for observation
            self.obs_item = pg.ImageItem()
            self.view.addItem(self.obs_item)

            # Add ROI rectangle for receptive field
            from pyqtgraph.Qt import QtCore, QtGui
            self.rf_box = pg.ROI([0, 0], [1, 1], pen=pg.mkPen(color='cyan', width=3), movable=False, resizable=False)
            # No need to remove handles - already has none when movable=False
            self.view.addItem(self.rf_box)

            # Add text label for RF info
            self.info_label = pg.TextItem(anchor=(0, 0), color='cyan')
            self.view.addItem(self.info_label)

            # Current RF box position (for center of activation map)
            self.current_box = None
            self.current_input = None

            self.is_open = True

            # Set up close event handler
            def on_close(event):
                self.is_open = False
                if _active_viewers.get('rf_overlay_viewer') is self:
                    _active_viewers['rf_overlay_viewer'] = None
                event.accept()

            self.win.closeEvent = on_close

            # Position window (use saved layout if available)
            layout = load_window_layout()
            if 'rf_overlay_viewer' in layout:
                pos = layout['rf_overlay_viewer']
                self.win.move(pos['x'], pos['y'])
            else:
                # Default position: below detail viewer
                self.win.move(800, 400)

            # Show window
            self.win.show()
            self.win.raise_()

            # Display RF info
            rf_size = rf_info['rf_size']
            self.info_label.setText(f"RF Size: {rf_size}×{rf_size}")
            self.info_label.setPos(5, 5)

        except ImportError:
            raise ImportError("PyQtGraph is required for receptive field overlay")

    def update_observation(self, obs, rf_box=None, act_y=None, act_x=None):
        """Update displayed observation and receptive field box.

        Args:
            obs: Input observation (numpy array, CHW or HW)
            rf_box: Dict with 'y_min', 'y_max', 'x_min', 'x_max' (optional)
            act_y: Activation Y coordinate (optional, for computing RF box)
            act_x: Activation X coordinate (optional, for computing RF box)
        """
        if not self.is_open:
            return

        self.current_input = obs

        # Render observation for display
        obs_rgb = self._render_obs(obs)
        if obs_rgb is not None:
            # Transpose for PyQtGraph
            if len(obs_rgb.shape) == 3:
                obs_rgb = self.np.transpose(obs_rgb, (1, 0, 2))
            elif len(obs_rgb.shape) == 2:
                obs_rgb = self.np.transpose(obs_rgb, (1, 0))

            self.obs_item.setImage(obs_rgb)

            # Get input shape (H, W)
            if len(obs.shape) == 3:
                input_h, input_w = obs.shape[1], obs.shape[2]
            else:
                input_h, input_w = obs.shape

            self.view.setRange(xRange=(0, input_h), yRange=(0, input_w), padding=0)

            # Compute RF box if activation coordinates provided
            if rf_box is None and act_y is not None and act_x is not None:
                rf_box = get_receptive_field_box(self.rf_info, act_y, act_x, (input_h, input_w))

            # Update RF box if provided
            if rf_box is not None:
                self.current_box = rf_box
                # PyQtGraph ROI uses (x, y, w, h) format
                x_min = rf_box['x_min']
                y_min = rf_box['y_min']
                width = rf_box['x_max'] - rf_box['x_min']
                height = rf_box['y_max'] - rf_box['y_min']

                # PyQtGraph ROI setPos expects [x, y] in view coords
                # With setRange(xRange=(0,H), yRange=(0,W)), this maps to [col, row] in original image
                self.rf_box.setPos([x_min, y_min])  # [x, y] = [col, row]
                self.rf_box.setSize([width, height])

            # Auto-size window
            scale = max(1, 400 // max(input_h, input_w))
            window_width = input_w * scale + 20
            window_height = input_h * scale + 50
            self.win.resize(int(window_width), int(window_height))

    def _render_obs(self, obs):
        """Render observation for display.

        Args:
            obs: Numpy array (C, H, W) or (H, W)

        Returns:
            RGB or grayscale array
        """
        if obs is None or obs.size == 0:
            return None

        # Normalize for display
        obs_min, obs_max = obs.min(), obs.max()
        if obs_max > obs_min:
            obs_norm = (obs - obs_min) / (obs_max - obs_min)
        else:
            obs_norm = self.np.zeros_like(obs)

        obs_uint8 = (obs_norm * 255).astype(self.np.uint8)

        # Handle different formats
        if len(obs.shape) == 2:
            # Grayscale (H, W)
            return obs_uint8
        elif len(obs.shape) == 3:
            if obs.shape[0] == 1:
                # Single channel (1, H, W) -> (H, W)
                return obs_uint8[0]
            elif obs.shape[0] == 3:
                # RGB (3, H, W) -> (H, W, 3)
                return self.np.transpose(obs_uint8, (1, 2, 0))
            else:
                # Multi-channel, average to grayscale
                return obs_uint8.mean(axis=0).astype(self.np.uint8)

        return obs_uint8

    def refresh(self, layer_idx=None, filter_idx=None):
        """Refresh with new filter or update current display.

        Args:
            layer_idx: New layer index (if changing filter)
            filter_idx: New filter index (if changing filter)
        """
        if not self.is_open:
            return

        # Update filter selection if provided
        if layer_idx is not None and filter_idx is not None:
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx
            self.win.setWindowTitle(f"Layer {layer_idx} Filter {filter_idx} - Receptive Field")

            # Get new RF info
            if self.parent_viewer:
                self.rf_info = self.parent_viewer.conv_info[layer_idx]['rf_info']

            # Update RF size label
            rf_size = self.rf_info['rf_size']
            self.info_label.setText(f"RF Size: {rf_size}×{rf_size}")

        # Refresh current observation if we have one
        if self.current_input is not None:
            # Re-compute RF box for center of activation map
            if self.current_box is not None:
                self.update_observation(self.current_input, rf_box=self.current_box)

    def close(self):
        """Close the viewer."""
        if self.is_open:
            try:
                self.win.close()
            except Exception:
                pass
            self.is_open = False
            if _active_viewers.get('rf_overlay_viewer') is self:
                _active_viewers['rf_overlay_viewer'] = None


class CNNFilterActivationDetailViewer:
    """Detail viewer for a single filter/activation pair."""

    def __init__(self, layer_idx, filter_idx, filter_data, activation_data=None, parent_viewer=None):
        """Initialize detail viewer for a single filter/activation.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index within the layer
            filter_data: Filter kernel data (2D numpy array)
            activation_data: Activation map data (2D numpy array, optional)
            parent_viewer: Parent CNNFilterActivationViewer instance for accessing colormap settings
        """
        try:
            import pyqtgraph as pg
            import numpy as np

            self.pg = pg
            self.np = np
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx
            self.parent_viewer = parent_viewer

            # Store data for re-rendering
            self.filter_data = filter_data
            self.activation_data = activation_data

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

            # Selection state for activation pixel
            self.selected_pos = None  # (act_y, act_x) or None
            self.selection_marker = None  # Visual marker for selected pixel

            # RF overlay view (only created if we have RF info and input)
            self.rf_view = None
            self.rf_obs_item = None
            self.rf_box = None
            self.rf_info = None
            if activation_data is not None and parent_viewer is not None:
                # Get RF info for this layer from conv_info
                if (hasattr(parent_viewer, 'conv_info') and
                    parent_viewer.conv_info and
                    layer_idx < len(parent_viewer.conv_info) and
                    'rf_info' in parent_viewer.conv_info[layer_idx]):
                    self.rf_info = parent_viewer.conv_info[layer_idx]['rf_info']

                    # Add RF view to the right
                    rf_view = self.win.addViewBox()
                    rf_view.setAspectLocked(True)
                    rf_view.invertY(True)
                    rf_view.setContentsMargins(0, 0, 0, 0)
                    rf_view.setMenuEnabled(False)
                    rf_view.setMouseEnabled(x=False, y=False)
                    rf_view.enableAutoRange(enable=False)
                    rf_obs_item = pg.ImageItem()
                    rf_view.addItem(rf_obs_item)

                    # Add RF box (initially hidden)
                    rf_box = pg.ROI([0, 0], [1, 1], pen=pg.mkPen(color='cyan', width=3),
                                   movable=False, resizable=False)
                    rf_view.addItem(rf_box)
                    rf_box.hide()

                    self.rf_view = rf_view
                    self.rf_obs_item = rf_obs_item
                    self.rf_box = rf_box

            # Add click handler to activation view for pixel selection
            if self.act_view is not None:
                self.act_view.scene().sigMouseClicked.connect(self._on_activation_clicked)

            # Render the data
            self._render(filter_data, activation_data)

            self.is_open = True

            # Set up close event handler
            def on_close(event):
                self.is_open = False
                # Unregister from global viewers
                if _active_viewers.get('cnn_detail_viewer') is self:
                    _active_viewers['cnn_detail_viewer'] = None
                event.accept()

            self.win.closeEvent = on_close

            # Show window
            self.win.show()
            self.win.raise_()

        except ImportError:
            raise ImportError("PyQtGraph is required for detail viewer")

    def _render(self, filter_data, activation_data):
        """Render filter and activation data."""
        # Store data for potential re-rendering
        self.filter_data = filter_data
        self.activation_data = activation_data

        # Apply diverging colormap to filter
        fh, fw = filter_data.shape
        filter_rgb = self._apply_diverging_colormap(filter_data)

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
            # Apply viridis colormap to activation
            ah, aw = activation_data.shape
            act_rgb = self._apply_viridis_colormap(activation_data)

            # Highlight selected pixel if any
            if self.selected_pos is not None:
                act_y, act_x = self.selected_pos
                if 0 <= act_y < ah and 0 <= act_x < aw:
                    # Use white to mark selection
                    act_rgb[act_y, act_x] = [255, 255, 255]  # White

            # Transpose for PyQtGraph
            act_rgb_t = self.np.transpose(act_rgb, (1, 0, 2))
            self.act_item.setImage(act_rgb_t)
            self.act_view.setRange(xRange=(0, ah), yRange=(0, aw), padding=0)

            # Add activation width to window
            window_width += ah * scale

        # Render RF overlay if view exists and we have input
        if self.rf_view is not None and self.parent_viewer and hasattr(self.parent_viewer, 'current_input'):
            input_obs = self.parent_viewer.current_input
            if input_obs is not None:
                # Render input observation
                obs_rgb = self._render_input_obs(input_obs)
                if obs_rgb is not None:
                    # Transpose for PyQtGraph
                    if len(obs_rgb.shape) == 3:
                        obs_rgb_t = self.np.transpose(obs_rgb, (1, 0, 2))
                    else:
                        obs_rgb_t = self.np.transpose(obs_rgb, (1, 0))
                    self.rf_obs_item.setImage(obs_rgb_t)

                    # Get input shape
                    if len(input_obs.shape) == 3:
                        input_h, input_w = input_obs.shape[1], input_obs.shape[2]
                    else:
                        input_h, input_w = input_obs.shape

                    self.rf_view.setRange(xRange=(0, input_h), yRange=(0, input_w), padding=0)

                    # Update RF box if pixel is selected
                    if self.selected_pos is not None:
                        act_y, act_x = self.selected_pos
                        rf_box_coords = get_receptive_field_box(self.rf_info, act_y, act_x, (input_h, input_w))
                        if rf_box_coords:
                            x_min = rf_box_coords['x_min']
                            y_min = rf_box_coords['y_min']
                            width = rf_box_coords['x_max'] - rf_box_coords['x_min']
                            height = rf_box_coords['y_max'] - rf_box_coords['y_min']
                            # PyQtGraph ROI coordinates: after transpose and setRange(xRange=(0,H), yRange=(0,W)),
                            # the view's x-axis spans [0,H] (rows) and y-axis spans [0,W] (cols).
                            # But ROI setPos expects [x, y] which should map to [col, row] in original space!
                            self.rf_box.setPos([x_min, y_min])  # [x, y] = [col, row]
                            self.rf_box.setSize([width, height])  # [width, height]
                            self.rf_box.show()
                    else:
                        self.rf_box.hide()

                    # Add RF view width to window - use fixed reasonable size
                    # RF input is typically 84x84, much larger than filter/activation
                    # Use a fixed scale that keeps it visible but not overwhelming
                    rf_display_width = 150  # Fixed reasonable width
                    window_width += rf_display_width

        self.win.resize(int(window_width), int(window_height) + 30)

    def _render_input_obs(self, obs):
        """Render input observation as RGB for display."""
        # Handle different observation formats
        if len(obs.shape) == 3:  # CHW image
            c, h, w = obs.shape
            if c == 1:  # Grayscale
                obs_2d = obs[0]
                obs_rgb = self.np.stack([obs_2d, obs_2d, obs_2d], axis=-1)
            elif c == 3:  # RGB
                obs_rgb = self.np.transpose(obs, (1, 2, 0))  # CHW -> HWC
            elif c == 4:  # Frame stacking - take most recent frame
                obs_2d = obs[-1]  # Most recent frame
                obs_rgb = self.np.stack([obs_2d, obs_2d, obs_2d], axis=-1)
            else:
                return None
        elif len(obs.shape) == 2:  # HW grayscale
            obs_rgb = self.np.stack([obs, obs, obs], axis=-1)
        else:
            return None

        # Normalize to 0-255 uint8
        obs_min, obs_max = obs_rgb.min(), obs_rgb.max()
        if obs_max > obs_min:
            obs_rgb = ((obs_rgb - obs_min) / (obs_max - obs_min) * 255).astype(self.np.uint8)
        else:
            obs_rgb = self.np.zeros_like(obs_rgb, dtype=self.np.uint8)

        return obs_rgb

    def _apply_diverging_colormap(self, data):
        """Apply diverging colormap to data (for filters).

        Uses parent viewer's colormap if available, otherwise defaults to RdBu.
        """
        # Use parent viewer's method if available
        if self.parent_viewer is not None:
            return self.parent_viewer._apply_diverging_colormap_to_array(data)

        # Fallback: Default RdBu implementation
        abs_max = max(abs(data.min()), abs(data.max()))
        if abs_max > 0:
            normalized = data / abs_max
        else:
            normalized = self.np.zeros_like(data)

        h, w = data.shape
        rgb = self.np.zeros((h, w, 3), dtype=self.np.uint8)

        neg_mask = normalized < 0
        pos_mask = normalized >= 0

        intensity_neg = self.np.abs(normalized[neg_mask])
        rgb[neg_mask, 0] = 255
        rgb[neg_mask, 1] = (255 * (1 - intensity_neg)).astype(self.np.uint8)
        rgb[neg_mask, 2] = (255 * (1 - intensity_neg)).astype(self.np.uint8)

        intensity_pos = normalized[pos_mask]
        rgb[pos_mask, 0] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
        rgb[pos_mask, 1] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
        rgb[pos_mask, 2] = 255

        return rgb

    def _apply_viridis_colormap(self, data):
        """Apply sequential colormap to data (for activations).

        Uses parent viewer's colormap if available, otherwise defaults to Viridis.
        """
        # Use parent viewer's method if available
        if self.parent_viewer is not None:
            return self.parent_viewer._apply_viridis_colormap_to_array(data)

        # Fallback: Default Viridis implementation
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            normalized = (data - d_min) / (d_max - d_min)
        else:
            normalized = self.np.zeros_like(data)

        viridis_values = self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0])
        viridis_colors = self.np.array([
            [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
            [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [253, 231, 37]
        ], dtype=self.np.float32)

        flat_normalized = normalized.ravel()
        indices = self.np.searchsorted(viridis_values, flat_normalized) - 1
        indices = self.np.clip(indices, 0, len(viridis_values) - 2)

        v0 = viridis_values[indices]
        v1 = viridis_values[indices + 1]
        c0 = viridis_colors[indices]
        c1 = viridis_colors[indices + 1]

        t = self.np.zeros_like(flat_normalized)
        valid_mask = v1 > v0
        t[valid_mask] = (flat_normalized[valid_mask] - v0[valid_mask]) / (v1[valid_mask] - v0[valid_mask])

        rgb_flat = c0 + t[:, self.np.newaxis] * (c1 - c0)
        rgb = rgb_flat.reshape(data.shape[0], data.shape[1], 3).astype(self.np.uint8)

        return rgb

    def refresh_colormaps(self):
        """Re-render with current colormap settings from parent."""
        if self.is_open and self.filter_data is not None:
            print(f"  DetailViewer: Refreshing colormaps for Layer {self.layer_idx} Filter {self.filter_idx}")
            self._render(self.filter_data, self.activation_data)

    def update_filter(self, layer_idx, filter_idx, filter_data, activation_data=None):
        """Update the viewer with a new filter/activation pair."""
        if not self.is_open:
            return

        self.layer_idx = layer_idx
        self.filter_idx = filter_idx

        # Update RF info if layer changed
        if (self.rf_view is not None and self.parent_viewer is not None and
            hasattr(self.parent_viewer, 'conv_info') and
            self.parent_viewer.conv_info and
            layer_idx < len(self.parent_viewer.conv_info) and
            'rf_info' in self.parent_viewer.conv_info[layer_idx]):
            self.rf_info = self.parent_viewer.conv_info[layer_idx]['rf_info']

        # Update window title
        self.win.setWindowTitle(f"Layer {layer_idx} Filter {filter_idx}")

        # Re-render with new data
        self._render(filter_data, activation_data)

    def update_activation(self, activation_data):
        """Update the activation display with new data."""
        if not self.is_open or self.act_item is None or activation_data is None:
            return

        # Store new activation data and re-render (preserves selection highlight and RF)
        self.activation_data = activation_data
        self._render(self.filter_data, activation_data)

    def _on_activation_clicked(self, event):
        """Handle clicks on activation map to toggle pixel selection."""
        if not self.act_view:
            return

        # Check if click is on activation view
        scene_pos = event.scenePos()
        if not self.act_view.sceneBoundingRect().contains(scene_pos):
            return

        # Map scene coordinates to view coordinates
        view_pos = self.act_view.mapSceneToView(scene_pos)

        # Convert to activation map coordinates (accounting for PyQtGraph transpose)
        # We transpose the image as (1, 0, 2): (H, W, C) -> (W, H, C)
        # After transpose: act_rgb_t[i, j] = act_rgb[j, i]
        # PyQtGraph displays act_rgb_t[i, j] at view position (x=i, y=j)
        # So view position (view_x, view_y) shows act_rgb_t[view_x, view_y] = act_rgb[view_y, view_x]
        # Since act_rgb[row, col], we have: view_x → col, view_y → row
        act_x = int(view_pos.x())  # view x → transposed index 0 → original column
        act_y = int(view_pos.y())  # view y → transposed index 1 → original row

        # Clamp to activation map bounds
        if self.activation_data is not None:
            h, w = self.activation_data.shape
            act_y = max(0, min(act_y, h - 1))
            act_x = max(0, min(act_x, w - 1))

        # Toggle selection
        if self.selected_pos == (act_y, act_x):
            # Clicked same pixel - deselect
            self.selected_pos = None
        else:
            # Select new pixel
            self.selected_pos = (act_y, act_x)

        # Re-render to update highlight and RF overlay
        self._render(self.filter_data, self.activation_data)

    def close(self):
        """Close the detail viewer."""
        if self.is_open:
            try:
                self.win.close()
            except Exception:
                pass
            self.is_open = False

            # Unregister from global viewers
            if _active_viewers.get('cnn_detail_viewer') is self:
                _active_viewers['cnn_detail_viewer'] = None


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

        # Colormap settings
        self.filter_colormap = 'rdbu'  # Default diverging colormap for filters
        self.activation_colormap = 'viridis'  # Default sequential colormap for activations

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

        # Calculate receptive field info for each layer
        rf_info_list = calculate_receptive_field_info(self.conv_layers)
        for idx, rf_info in enumerate(rf_info_list):
            self.conv_info[idx]['rf_info'] = rf_info

        # Track single detail viewer (reused for all clicks)
        self.detail_viewer = None

        # Track receptive field overlay viewer
        self.rf_overlay_viewer = None

        # Track single maximal activation viewer (synced with detail viewer)
        self.maximal_activation_viewer = None

        # Initialize activation buffers for each filter
        # Structure: self.activation_buffers[layer_idx][filter_idx] = TopKActivationBuffer
        self.activation_buffers = []
        for info in self.conv_info:
            layer_buffers = [TopKActivationBuffer(k=10) for _ in range(info['out_channels'])]
            self.activation_buffers.append(layer_buffers)

        # Store current input observation for buffer updates
        self.current_input = None

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

            # Update activation buffers if we have input
            if self.current_input is not None:
                try:
                    # Output shape: (batch, channels, H, W)
                    output_cpu = output.detach().cpu()
                    if output_cpu.dim() == 4:
                        output_cpu = output_cpu[0]  # Take first batch element

                    # For each filter/channel, compute max activation and update buffer
                    n_channels = output_cpu.shape[0]
                    for filter_idx in range(n_channels):
                        activation_map = output_cpu[filter_idx]  # (H, W)
                        max_activation = float(activation_map.max())

                        # Update buffer with input patch and max activation
                        if self.current_input is not None:
                            self.activation_buffers[layer_idx][filter_idx].update(
                                max_activation,
                                self.current_input
                            )
                except Exception as e:
                    # Don't let buffer updates crash the forward pass
                    import sys
                    print(f"Warning: Failed to update activation buffer: {e}", file=sys.stderr)

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

    def _open_maximal_activation_viewer(self, layer_idx, filter_idx):
        """Open maximal activation viewer for a specific filter.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index
        """
        try:
            # Get buffer for this filter
            buffer = self.get_buffer(layer_idx, filter_idx)
            if buffer is None:
                print(f"No buffer found for Layer {layer_idx} Filter {filter_idx}")
                return

            # Close existing maximal activation viewer if open
            if _active_viewers.get('maximal_activation_viewer') is not None:
                try:
                    _active_viewers['maximal_activation_viewer'].close()
                except Exception:
                    pass

            # Create new viewer
            viewer = MaximalActivationViewer(layer_idx, filter_idx, buffer, parent_viewer=self)
            _active_viewers['maximal_activation_viewer'] = viewer

            print(f"Opened maximal activation viewer: Layer {layer_idx} Filter {filter_idx}")
        except Exception as e:
            import sys
            print(f"Error opening maximal activation viewer: {e}", file=sys.stderr)

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
                    layer_idx, filter_idx, filter_2d, activation_2d, parent_viewer=self
                )

                # Register in global viewers for F9 hotkey
                _active_viewers['cnn_detail_viewer'] = self.detail_viewer

                # Position window (use saved layout if available)
                layout = load_window_layout()
                if 'cnn_detail_viewer' in layout:
                    pos = layout['cnn_detail_viewer']
                    self.detail_viewer.win.move(pos['x'], pos['y'])
                    if pos['width'] and pos['height']:
                        self.detail_viewer.win.resize(pos['width'], pos['height'])
                else:
                    # Default position
                    self.detail_viewer.win.move(800, 100)

                # Install keyboard shortcuts on detail viewer
                install_keyboard_shortcuts(self.detail_viewer.win)

                print(f"Opened detail viewer: Layer {layer_idx} Filter {filter_idx}")
            else:
                # Update existing viewer with new filter/activation
                self.detail_viewer.update_filter(layer_idx, filter_idx, filter_2d, activation_2d)
                self.detail_viewer.win.raise_()  # Bring to front
                print(f"Updated detail viewer: Layer {layer_idx} Filter {filter_idx}")

            # Auto-open or update maximal activation viewer (stays in sync with detail viewer)
            self._sync_maximal_activation_viewer(layer_idx, filter_idx)

            # RF overlay is now integrated into detail viewer (click activation pixels to select)
            # self._sync_rf_overlay_viewer(layer_idx, filter_idx)

        except Exception as e:
            import sys
            print(f"Error with detail viewer: {e}", file=sys.stderr)

    def _sync_rf_overlay_viewer(self, layer_idx, filter_idx):
        """Keep receptive field overlay in sync with detail viewer.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index
        """
        try:
            # Get RF info for this layer
            rf_info = self.conv_info[layer_idx]['rf_info']

            if self.rf_overlay_viewer is None or not self.rf_overlay_viewer.is_open:
                # Create new RF overlay viewer
                self.rf_overlay_viewer = ReceptiveFieldOverlay(
                    layer_idx, filter_idx, rf_info, parent_viewer=self
                )
                _active_viewers['rf_overlay_viewer'] = self.rf_overlay_viewer

                # Initialize with current observation if available
                if self.current_input is not None:
                    # Show RF box for center of activation map
                    activation = self.conv_info[layer_idx].get('activation')
                    if activation is not None:
                        if activation.dim() == 4:
                            activation = activation[0]
                        # Use center of activation map
                        act_h, act_w = activation.shape[1], activation.shape[2]
                        center_y, center_x = act_h // 2, act_w // 2
                        self.rf_overlay_viewer.update_observation(
                            self.current_input, act_y=center_y, act_x=center_x
                        )
            else:
                # Update existing viewer to new filter
                self.rf_overlay_viewer.refresh(layer_idx, filter_idx)

                # Update with current observation
                if self.current_input is not None:
                    activation = self.conv_info[layer_idx].get('activation')
                    if activation is not None:
                        if activation.dim() == 4:
                            activation = activation[0]
                        act_h, act_w = activation.shape[1], activation.shape[2]
                        center_y, center_x = act_h // 2, act_w // 2
                        self.rf_overlay_viewer.update_observation(
                            self.current_input, act_y=center_y, act_x=center_x
                        )

        except Exception as e:
            import sys
            print(f"Error syncing RF overlay viewer: {e}", file=sys.stderr)

    def _sync_maximal_activation_viewer(self, layer_idx, filter_idx):
        """Keep maximal activation viewer in sync with detail viewer.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index
        """
        try:
            # Get buffer for this filter
            buffer = self.get_buffer(layer_idx, filter_idx)
            if buffer is None:
                return

            if self.maximal_activation_viewer is None or not self.maximal_activation_viewer.is_open:
                # Create new maximal activation viewer
                self.maximal_activation_viewer = MaximalActivationViewer(
                    layer_idx, filter_idx, buffer, parent_viewer=self
                )
                _active_viewers['maximal_activation_viewer'] = self.maximal_activation_viewer
            else:
                # Update existing viewer to new filter
                self.maximal_activation_viewer.refresh(layer_idx, filter_idx)

        except Exception as e:
            import sys
            print(f"Error syncing maximal activation viewer: {e}", file=sys.stderr)

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
                # Filter grid is already uint8 RGB
                # PyQtGraph expects (W, H, 3) format for RGB images, but numpy creates (H, W, 3)
                # Transpose spatial dimensions: (H, W, 3) -> (W, H, 3)
                filter_grid_transposed = self.np.transpose(filter_grid, (1, 0, 2))
                self.filter_items[idx].setImage(filter_grid_transposed)

                # Set view range to match transposed dimensions
                h, w = filter_grid.shape[:2]
                self.filter_views[idx].setRange(xRange=(0, h), yRange=(0, w), padding=0)
            else:
                self.axes[idx, 0].clear()
                # Display RGB image (no colormap)
                self.axes[idx, 0].imshow(filter_grid)
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

        # Create RGB grid (start with dark gray border: R=30, G=30, B=30)
        grid = self.np.full((grid_h, grid_w, 3), 30, dtype=self.np.uint8)

        for i in range(out_ch):
            row = i // grid_cols
            col = i % grid_cols

            # Add padding offset to position each filter
            y_start = row * (kH + padding) + padding
            x_start = col * (kW + padding) + padding

            # Apply diverging colormap to filter
            filt = filters_2d[i]
            filt_rgb = self._apply_diverging_colormap_to_array(filt)

            # Place colored filter
            grid[y_start:y_start + kH, x_start:x_start + kW] = filt_rgb

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

        # Create RGB grid (start with dark gray border: R=30, G=30, B=30)
        grid = self.np.full((grid_h, grid_w, 3), 30, dtype=self.np.uint8)

        for i in range(n_channels):
            row = i // grid_cols
            col = i % grid_cols

            # Add padding offset to position each activation map
            y_start = row * (H + padding) + padding
            x_start = col * (W + padding) + padding

            # Apply viridis colormap to activation
            act = act_np[i]
            act_rgb = self._apply_viridis_colormap_to_array(act)

            # Place colored activation
            grid[y_start:y_start + H, x_start:x_start + W] = act_rgb

        return grid

    def _apply_diverging_colormap_to_array(self, data):
        """Apply diverging colormap to data (for filters).

        Uses self.filter_colormap to determine which colormap to apply.
        """
        # Normalize to [-1, 1] range centered at zero
        abs_max = max(abs(data.min()), abs(data.max()))
        if abs_max > 0:
            normalized = data / abs_max  # Range: [-1, 1]
        else:
            normalized = self.np.zeros_like(data)

        h, w = data.shape
        rgb = self.np.zeros((h, w, 3), dtype=self.np.uint8)

        # Map normalized values from [-1, 1] to [0, 1] for colormap lookup
        normalized_01 = (normalized + 1) / 2

        # Select colormap
        if self.filter_colormap == 'rdbu':
            # Red-Blue diverging (red=negative, white=zero, blue=positive)
            neg_mask = normalized < 0
            pos_mask = normalized >= 0
            intensity_neg = self.np.abs(normalized[neg_mask])
            rgb[neg_mask, 0] = 255
            rgb[neg_mask, 1] = (255 * (1 - intensity_neg)).astype(self.np.uint8)
            rgb[neg_mask, 2] = (255 * (1 - intensity_neg)).astype(self.np.uint8)
            intensity_pos = normalized[pos_mask]
            rgb[pos_mask, 0] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
            rgb[pos_mask, 1] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
            rgb[pos_mask, 2] = 255
        elif self.filter_colormap == 'rdylbu':
            # Red-Yellow-Blue diverging
            rgb = self._apply_sequential_colormap(normalized_01, 'rdylbu')
        elif self.filter_colormap == 'seismic':
            # Seismic colormap (blue-white-red)
            rgb = self._apply_sequential_colormap(normalized_01, 'seismic')
        elif self.filter_colormap == 'coolwarm':
            # Coolwarm (blue-white-red)
            rgb = self._apply_sequential_colormap(normalized_01, 'coolwarm')
        else:
            # Default to RdBu
            neg_mask = normalized < 0
            pos_mask = normalized >= 0
            intensity_neg = self.np.abs(normalized[neg_mask])
            rgb[neg_mask, 0] = 255
            rgb[neg_mask, 1] = (255 * (1 - intensity_neg)).astype(self.np.uint8)
            rgb[neg_mask, 2] = (255 * (1 - intensity_neg)).astype(self.np.uint8)
            intensity_pos = normalized[pos_mask]
            rgb[pos_mask, 0] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
            rgb[pos_mask, 1] = (255 * (1 - intensity_pos)).astype(self.np.uint8)
            rgb[pos_mask, 2] = 255

        return rgb

    def _apply_viridis_colormap_to_array(self, data):
        """Apply sequential colormap to data (for activations).

        Uses self.activation_colormap to determine which colormap to apply.
        """
        # Normalize to [0, 1]
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            normalized = (data - d_min) / (d_max - d_min)
        else:
            normalized = self.np.zeros_like(data)

        return self._apply_sequential_colormap(normalized, self.activation_colormap)

    def _apply_sequential_colormap(self, normalized, colormap_name):
        """Apply a sequential or diverging colormap to normalized data [0, 1].

        Args:
            normalized: Normalized data in range [0, 1]
            colormap_name: Name of the colormap

        Returns:
            RGB array (H, W, 3)
        """
        # Define colormap lookup tables (sampled at key points)
        colormaps = {
            'viridis': {
                'values': self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0]),
                'colors': self.np.array([
                    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
                    [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [253, 231, 37]
                ], dtype=self.np.float32)
            },
            'plasma': {
                'values': self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0]),
                'colors': self.np.array([
                    [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150],
                    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33]
                ], dtype=self.np.float32)
            },
            'inferno': {
                'values': self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0]),
                'colors': self.np.array([
                    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
                    [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]
                ], dtype=self.np.float32)
            },
            'magma': {
                'values': self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0]),
                'colors': self.np.array([
                    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
                    [181, 54, 122], [229, 80, 100], [251, 136, 97], [254, 194, 135], [252, 253, 191]
                ], dtype=self.np.float32)
            },
            'cividis': {
                'values': self.np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0]),
                'colors': self.np.array([
                    [0, 32, 76], [0, 50, 85], [30, 66, 88], [54, 81, 94],
                    [79, 97, 107], [109, 114, 121], [141, 132, 140], [178, 153, 164], [231, 179, 195]
                ], dtype=self.np.float32)
            },
            'rdylbu': {
                'values': self.np.array([0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]),
                'colors': self.np.array([
                    [165, 0, 38], [244, 109, 67], [254, 224, 144],
                    [255, 255, 191], [224, 243, 248], [116, 173, 209], [49, 54, 149]
                ], dtype=self.np.float32)
            },
            'seismic': {
                'values': self.np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                'colors': self.np.array([
                    [0, 0, 76], [67, 133, 255], [255, 255, 255], [255, 100, 0], [127, 0, 0]
                ], dtype=self.np.float32)
            },
            'coolwarm': {
                'values': self.np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                'colors': self.np.array([
                    [59, 76, 192], [144, 178, 254], [221, 221, 221], [245, 156, 125], [180, 4, 38]
                ], dtype=self.np.float32)
            }
        }

        # Get colormap or default to viridis
        cmap = colormaps.get(colormap_name, colormaps['viridis'])
        cmap_values = cmap['values']
        cmap_colors = cmap['colors']

        # Flatten normalized array for vectorized lookup
        flat_normalized = normalized.ravel()

        # Find the indices of the color stops for each value
        indices = self.np.searchsorted(cmap_values, flat_normalized) - 1
        indices = self.np.clip(indices, 0, len(cmap_values) - 2)

        # Get the two color stops for interpolation
        v0 = cmap_values[indices]
        v1 = cmap_values[indices + 1]
        c0 = cmap_colors[indices]
        c1 = cmap_colors[indices + 1]

        # Compute interpolation factor
        t = self.np.zeros_like(flat_normalized)
        valid_mask = v1 > v0
        t[valid_mask] = (flat_normalized[valid_mask] - v0[valid_mask]) / (v1[valid_mask] - v0[valid_mask])

        # Linear interpolation: c0 + t * (c1 - c0)
        rgb_flat = c0 + t[:, self.np.newaxis] * (c1 - c0)
        rgb = rgb_flat.reshape(normalized.shape[0], normalized.shape[1], 3).astype(self.np.uint8)

        return rgb

    def set_input(self, obs):
        """Store current input observation for activation buffer updates.

        Args:
            obs: Input observation (numpy array or tensor), typically (C, H, W) or (H, W)
        """
        import torch
        import numpy as np

        # Convert to numpy if tensor
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        # Store copy to avoid reference issues
        self.current_input = obs.copy() if isinstance(obs, np.ndarray) else obs

    def get_buffer(self, layer_idx, filter_idx):
        """Get the activation buffer for a specific filter.

        Args:
            layer_idx: Layer index
            filter_idx: Filter index within the layer

        Returns:
            TopKActivationBuffer instance
        """
        if layer_idx < len(self.activation_buffers) and filter_idx < len(self.activation_buffers[layer_idx]):
            return self.activation_buffers[layer_idx][filter_idx]
        return None

    def save_activation_buffers(self, filepath):
        """Save activation buffers to disk for offline analysis.

        Args:
            filepath: Path to save file (will create .npz file)
        """
        import numpy as np
        from pathlib import Path

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Collect all buffer data
            buffer_data = {}
            for layer_idx, layer_buffers in enumerate(self.activation_buffers):
                for filter_idx, buffer in enumerate(layer_buffers):
                    top_k = buffer.get_top_k()
                    if top_k:
                        # Store activation values and patches separately
                        key_prefix = f"layer{layer_idx}_filter{filter_idx}"
                        activations = np.array([val for val, _ in top_k])
                        patches = np.array([patch for _, patch in top_k])
                        buffer_data[f"{key_prefix}_activations"] = activations
                        buffer_data[f"{key_prefix}_patches"] = patches

            # Save to compressed npz file
            np.savez_compressed(filepath, **buffer_data)
            print(f"Saved activation buffers to {filepath}")

        except Exception as e:
            import sys
            print(f"Error saving activation buffers: {e}", file=sys.stderr)

    def load_activation_buffers(self, filepath):
        """Load activation buffers from disk.

        Args:
            filepath: Path to load file (.npz file)
        """
        import numpy as np
        from pathlib import Path

        try:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"File not found: {filepath}")
                return

            # Load data
            data = np.load(filepath)

            # Reconstruct buffers
            for layer_idx, layer_buffers in enumerate(self.activation_buffers):
                for filter_idx, buffer in enumerate(layer_buffers):
                    key_prefix = f"layer{layer_idx}_filter{filter_idx}"
                    act_key = f"{key_prefix}_activations"
                    patch_key = f"{key_prefix}_patches"

                    if act_key in data and patch_key in data:
                        # Clear existing buffer
                        buffer.clear()

                        # Reload data
                        activations = data[act_key]
                        patches = data[patch_key]

                        for act_val, patch in zip(activations, patches):
                            buffer.update(float(act_val), patch)

            print(f"Loaded activation buffers from {filepath}")

        except Exception as e:
            import sys
            print(f"Error loading activation buffers: {e}", file=sys.stderr)

    def clear_all_buffers(self):
        """Clear all activation buffers."""
        for layer_buffers in self.activation_buffers:
            for buffer in layer_buffers:
                buffer.clear()
        print("Cleared all activation buffers")

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
                    # Activation grid is already uint8 RGB
                    # PyQtGraph expects (W, H, 3) format for RGB images, but numpy creates (H, W, 3)
                    # Transpose spatial dimensions: (H, W, 3) -> (W, H, 3)
                    act_grid_transposed = self.np.transpose(act_grid, (1, 0, 2))
                    self.activation_items[idx].setImage(act_grid_transposed)

                    # Set view range to match transposed dimensions
                    h, w = act_grid.shape[:2]
                    self.activation_views[idx].setRange(xRange=(0, h), yRange=(0, w), padding=0)
                else:
                    self.axes[idx, 1].clear()
                    # Display RGB image (no colormap)
                    self.axes[idx, 1].imshow(act_grid)
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

            # Update maximal activation viewer with new buffer data (real-time refresh)
            if self.maximal_activation_viewer is not None and self.maximal_activation_viewer.is_open:
                # Refresh without changing filter - just update images with latest top-K
                self.maximal_activation_viewer.refresh()

            # RF overlay is now integrated into detail viewer - no separate window to update
            # (detail viewer's _render() automatically updates RF when activation changes)

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

    def set_filter_colormap(self, colormap_name):
        """Change the filter colormap and re-render filters.

        Args:
            colormap_name: Name of colormap ('RdBu', 'RdYlBu', 'Seismic', 'Coolwarm')
        """
        print(f"  CNNViewer: set_filter_colormap called with '{colormap_name}', is_open={self.is_open}")

        if not self.is_open:
            print(f"  CNNViewer: Viewer is closed, skipping colormap change")
            return

        old_colormap = self.filter_colormap
        self.filter_colormap = colormap_name.lower()
        print(f"  CNNViewer: Changed filter colormap from '{old_colormap}' to '{self.filter_colormap}'")

        # Re-render filters with new colormap
        try:
            print(f"  CNNViewer: Calling _render_filters()...")
            self._render_filters()
            print(f"  CNNViewer: _render_filters() completed")

            # Force Qt event processing to update display
            if self.use_pyqtgraph:
                try:
                    from pyqtgraph.Qt import QtWidgets
                    QtWidgets.QApplication.processEvents()
                    print(f"  CNNViewer: Qt events processed")
                except Exception as e:
                    print(f"  CNNViewer: Error processing Qt events: {e}")

            # Refresh detail viewer if open
            if self.detail_viewer is not None and self.detail_viewer.is_open:
                print(f"  CNNViewer: Refreshing detail viewer with new colormap...")
                self.detail_viewer.refresh_colormaps()
        except Exception as e:
            print(f"  CNNViewer: Error rendering filters: {e}")
            import traceback
            traceback.print_exc()

    def set_activation_colormap(self, colormap_name):
        """Change the activation colormap and re-render activations.

        Args:
            colormap_name: Name of colormap ('Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis')
        """
        print(f"  CNNViewer: set_activation_colormap called with '{colormap_name}', is_open={self.is_open}")

        if not self.is_open:
            print(f"  CNNViewer: Viewer is closed, skipping colormap change")
            return

        old_colormap = self.activation_colormap
        self.activation_colormap = colormap_name.lower()
        print(f"  CNNViewer: Changed activation colormap from '{old_colormap}' to '{self.activation_colormap}'")

        # Force re-render of activations if they exist
        try:
            if self.use_pyqtgraph:
                print(f"  CNNViewer: Calling _update_activations()...")
                self._update_activations()
                print(f"  CNNViewer: _update_activations() completed")

                # Force Qt event processing to update display
                try:
                    from pyqtgraph.Qt import QtWidgets
                    QtWidgets.QApplication.processEvents()
                    print(f"  CNNViewer: Qt events processed")
                except Exception as e:
                    print(f"  CNNViewer: Error processing Qt events: {e}")

                # Refresh detail viewer if open
                if self.detail_viewer is not None and self.detail_viewer.is_open:
                    print(f"  CNNViewer: Refreshing detail viewer with new colormap...")
                    self.detail_viewer.refresh_colormaps()
        except Exception as e:
            print(f"  CNNViewer: Error rendering activations: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Close the viewer and remove hooks."""
        if not self.is_open:
            return

        # Mark as closed immediately to prevent race conditions
        self.is_open = False

        # Remove forward hooks
        for handle in self.activation_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.activation_handles.clear()

        # Close detail viewer if open
        if self.detail_viewer is not None:
            try:
                self.detail_viewer.close()
            except Exception:
                pass
            finally:
                self.detail_viewer = None

        # Close the main window
        try:
            if self.use_pyqtgraph:
                if hasattr(self, 'win') and self.win is not None:
                    self.win.close()
                    # Force window deletion
                    try:
                        from pyqtgraph.Qt import QtWidgets
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
            else:
                if hasattr(self, 'fig') and self.fig is not None:
                    self.plt.close(self.fig)
        except Exception as e:
            print(f"Warning: Error closing CNN viewer window: {e}")


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

    # Ensure keys are integers (JSON serialization may convert them to strings)
    original_labels = {int(k): v for k, v in original_labels.items()}

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


def play_episodes_manual(env, target_episodes: int, mode: str, step_by_step: bool = False, fps: int | None = None, action_labels: dict[int, str] | None = None, plotter: RewardPlotter | None = None, obs_viewer: PreprocessedObservationViewer | None = None, action_viewer: ActionVisualizer | None = None):
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

        # Update action visualizer
        if action_viewer and action_viewer.is_open:
            if is_multibinary:
                # For MultiBinary: show button states (0.0 or 1.0)
                button_states = [float(action[i]) for i in range(len(action))]
                action_viewer.update(button_states, None)
            else:
                # For Discrete: show uniform distribution with selected action highlighted
                # (we don't have true probabilities in manual mode)
                uniform_probs = [1.0 / n_actions] * n_actions
                action_viewer.update(uniform_probs, action)

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
        help="Action mode: 'trained' (use trained policy), 'random' (use untrained/random policy), 'user' (keyboard input). Default: 'trained' for --run-id, 'random' for --config-id",
    )
    p.add_argument("--seed", type=str, default=None, help="Random seed for environment (int, 'train', 'val', 'test', or None for test seed)")
    p.add_argument("--fps", type=int, default=None, help="Limit playback to target FPS (frames per second)")
    p.add_argument("--plot-metrics", dest="plot_metrics", action="store_true", default=True, help="Show real-time reward plot (default: True)")
    p.add_argument("--no-plot-metrics", dest="plot_metrics", action="store_false", help="Disable real-time reward plot")
    p.add_argument("--plot-update-interval", type=float, default=0.2, help="Minimum time (seconds) between plot updates (default: 0.2, lower=smoother but slower game)")
    p.add_argument("--plot-window-size", type=int, default=100, help="Number of steps to show in sliding window (default: 100)")
    p.add_argument("--show-preprocessing", dest="show_preprocessing", action="store_true", default=False, help="Show preprocessed observations in separate window (what agent sees)")
    p.add_argument("--show-actions", dest="show_actions", action="store_true", default=False, help="Show action probabilities in separate window (requires trained policy)")
    p.add_argument("--show-cnn-filters", dest="show_cnn_filters", action="store_true", default=False, help="Show CNN filters and activations in separate window (requires CNN policy)")
    p.add_argument("--show-obs", dest="show_obs", action="store_true", default=None, help="Show live observation values during playback (default: True for interactive, False when headless)")
    p.add_argument("--no-show-obs", dest="show_obs", action="store_false", help="Disable live observation values")
    p.add_argument("--toolbar", action="store_true", default=True, help="Show interactive visualization toolbar for toggling viewers and changing colormaps (default: True)")
    p.add_argument("--no-toolbar", dest="toolbar", action="store_false", help="Disable interactive visualization toolbar")
    p.add_argument(
        "--env-kwargs",
        action="append",
        dest="env_kwargs",
        metavar="KEY=VALUE",
        help="Override env_kwargs fields (e.g., --env-kwargs state=Level2-1). Can be specified multiple times.",
    )
    args = p.parse_args()
    target_episodes = max(1, int(args.episodes))

    # Default show_obs based on headless mode if not explicitly set
    if args.show_obs is None:
        args.show_obs = not args.headless

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
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=args.show_obs, target_episodes=target_episodes)

    # Extract action labels from config
    action_labels = extract_action_labels_from_config(config)
    if action_labels:
        print(f"Loaded {len(action_labels)} action labels from config spec")
    else:
        print("No action labels found in config spec")

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

    # Initialize action visualizer if enabled and not headless
    action_viewer = None
    if args.show_actions and not args.headless:
        try:
            # Get number of actions from environment
            from gymnasium.spaces import Discrete, MultiBinary
            action_space = env.single_action_space
            if isinstance(action_space, (Discrete, MultiBinary)):
                n_actions = action_space.n
                action_viewer = ActionVisualizer(n_actions=n_actions, action_labels=action_labels, update_interval=0.05)
                _active_viewers['action_visualizer'] = action_viewer
                backend = "PyQtGraph (fast)" if action_viewer.use_pyqtgraph else "Matplotlib"
                space_type = "MultiBinary buttons" if isinstance(action_space, MultiBinary) else "Discrete actions"
                print(f"Action visualizer enabled using {backend} for {space_type}")
                print("Close the viewer window to disable it.")
            else:
                print(f"Warning: Action visualizer only supports Discrete and MultiBinary action spaces (got {type(action_space).__name__})")
        except Exception as e:
            print(f"Warning: Could not initialize action visualizer: {e}")
            action_viewer = None

    # Handle different modes
    if args.mode == "user":
        # User control mode doesn't need policy
        try:
            play_episodes_manual(env, target_episodes, args.mode, args.step_by_step, args.fps, action_labels, plotter, obs_viewer, action_viewer)
        finally:
            if plotter:
                plotter.close()
            if obs_viewer:
                obs_viewer.close()
            if action_viewer:
                action_viewer.close()
        print("Done.")
        return

    # Load or build policy based on mode
    if args.mode == "trained":
        # Trained mode: load policy from checkpoint
        assert run is not None, "run must be loaded for trained mode"
        # TODO: we should be loading the agent and having it run the episode
        policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)
    elif args.mode == "random":
        # Random mode: build new untrained policy
        from utils.policy_factory import build_policy_from_env_and_config
        policy_model = build_policy_from_env_and_config(env, config)
        print(f"Initialized random policy: {policy_model.__class__.__name__}")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

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

    # Initialize visualization toolbar if enabled (after viewers are created)
    toolbar = None
    if args.toolbar and not args.headless:
        try:
            toolbar = VisualizationToolbar(config, policy_model, env, reward_plotter=plotter, obs_viewer=obs_viewer, action_viewer=action_viewer, cnn_viewer=cnn_viewer)
            print("Visualization toolbar enabled")
            print("  Click buttons to toggle viewers and change colormaps")
            print("  Hotkeys: [R] Reward | [O] Observation | [A] Actions | [F] Filters | [Q] Close")
        except Exception as e:
            print(f"Warning: Could not initialize visualization toolbar: {e}")
            toolbar = None

    # Initialize rollout collector; step-by-step mode, FPS limiting, plotting, or viewers use single-step rollouts
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1 if (args.step_by_step or args.fps or plotter or obs_viewer or action_viewer or cnn_viewer) else config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    # Print hotkey instructions if any viewers are active
    if plotter or obs_viewer or action_viewer or cnn_viewer:
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
                # Pass current observation to CNN viewer for activation buffer updates
                if collector.n_steps == 1:
                    obs = collector._buffer.obs_buf[collector._buffer.pos - 1, 0]
                    cnn_viewer.set_input(obs)
                cnn_viewer.update()

            # Update action visualizer if enabled
            if action_viewer and action_viewer.is_open and collector.n_steps == 1:
                from gymnasium.spaces import MultiBinary
                action_space = env.single_action_space

                if isinstance(action_space, MultiBinary):
                    # For MultiBinary: show button states (0.0 or 1.0)
                    action_array = collector._buffer.actions_buf[collector._buffer.pos - 1, 0]
                    button_states = [float(action_array[i]) for i in range(len(action_array))]
                    # No single "selected" action for MultiBinary, pass None
                    action_viewer.update(button_states, None)
                else:
                    # For Discrete: show action probabilities
                    action_probs = env.get_action_probs()
                    # Get the selected action from the buffer
                    selected_action = int(collector._buffer.actions_buf[collector._buffer.pos - 1, 0])
                    # Update action visualizer
                    if action_probs is not None:
                        action_viewer.update(action_probs[0], selected_action)

            # Update toolbar if enabled
            if toolbar and toolbar.is_open:
                # Handle toolbar events and get updated viewer references
                result = toolbar.handle_events()
                if result is not None:
                    toolbar_plotter, toolbar_obs, toolbar_action, toolbar_cnn = result

                    # Update viewer references from toolbar
                    if toolbar_plotter is not None:
                        plotter = toolbar_plotter
                    if toolbar_obs is not None:
                        obs_viewer = toolbar_obs
                    if toolbar_action is not None:
                        action_viewer = toolbar_action
                    if toolbar_cnn is not None:
                        cnn_viewer = toolbar_cnn

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
        if toolbar:
            toolbar.close()
        if plotter:
            plotter.close()
        if obs_viewer:
            obs_viewer.close()
        if action_viewer:
            action_viewer.close()
        if cnn_viewer:
            cnn_viewer.close()

    print("Done.")


if __name__ == "__main__":
    main()
