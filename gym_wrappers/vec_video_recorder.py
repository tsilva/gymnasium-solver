### 
# DISCLAIMER: originally from stable_baselines3/common/vec_env/vec_video_recorder.py
###

import os
import os.path
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
from gymnasium import error, logger
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    Note: for now it only allows to record one video and all videos
    must have at least two frames.

    The video recorder code was adapted from Gymnasium v1.0.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    :param enable_overlay: Whether to add episode and step overlay text to frames
    :param font_size: Size of the overlay text font
    :param text_position: (x, y) position for the overlay text
    :param text_color: RGB color tuple for the text
    :param stroke_color: RGB color tuple for the text outline
    :param stroke_width: Width of the text outline in pixels
    :param record_env_idx: If set, record only this env index from a VecEnv (defaults to 0).
                           When None, falls back to the default VecEnv tiled render of all envs.
    """

    def __init__(
        self,
        venv: VecEnv,
        video_length: Optional[int] = 200,
    record_env_idx: Optional[int] = 0,
        # Text overlay options
        enable_overlay: bool = True,
        font_size: int = 12,
        text_position: Tuple[int, int] = (6, 6),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        stroke_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_width: int = 2,
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:  # pragma: no cover # assume gym interface
            metadata = temp_env.metadata

        self.env.metadata = metadata
        assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

        self.frames_per_sec = self.env.metadata.get("render_fps", 30)

        self.step_id = 0
        self.video_length = video_length
        self.record_env_idx = record_env_idx
        try:
            self.num_envs = getattr(self.env, "num_envs", None)
        except Exception:
            self.num_envs = None

        self.recording = False
        self.recorded_frames: list[np.ndarray] = []
        
        # Episode and step tracking for overlay
        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0
        
        # Overlay configuration
        self.enable_overlay = enable_overlay
        self.font_size = font_size
        self.text_position = text_position
        self.text_color = text_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self._font = None  # Will be initialized when needed

        try:
            import moviepy  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise error.DependencyNotInstalled("MoviePy is not installed, run `pip install 'gymnasium[other]'`") from e

    def _get_font(self):
        """Get or create the font for text overlay."""
        if self._font is None:
            try:
                from PIL import ImageFont
                try:
                    # Try to load a monospace font
                    self._font = ImageFont.truetype("DejaVuSansMono.ttf", size=self.font_size)
                except OSError:
                    try:
                        # Fallback to Arial or similar
                        self._font = ImageFont.truetype("Arial.ttf", size=self.font_size)
                    except OSError:
                        # Use default font
                        self._font = ImageFont.load_default()
            except ImportError:
                # PIL not available, overlay will be disabled
                self.enable_overlay = False
                self._font = None
        return self._font

    def _add_overlay_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add episode, step and reward overlay to the frame.

        The overlay is rendered as small, vertically stacked, high-contrast text
        anchored to the top-left corner for maximum legibility.
        """
        if not self.enable_overlay:
            return frame
            
        try:
            from PIL import Image, ImageDraw
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            
            # Create overlay text (stacked vertically)
            lines = [
                f"Episode: {self.current_episode + 1}",
                f"Step: {self.current_step}",
                f"Reward: {self.accumulated_reward:.2f}",
            ]
            
            # Get font
            font = self._get_font()
            
            if font is not None:
                x, y = self.text_position

                # Measure text block to draw a solid high-contrast background rectangle
                try:
                    # Preferred precise measurement
                    bboxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
                    line_heights = [(b[3] - b[1]) for b in bboxes]
                    line_widths = [(b[2] - b[0]) for b in bboxes]
                except Exception:
                    # Fallback sizing if textbbox is not available
                    line_widths, line_heights = [], []
                    for t in lines:
                        w, h = font.getsize(t)
                        line_widths.append(w)
                        line_heights.append(h)

                line_spacing = max(1, int(self.font_size * 0.25))
                padding = 4
                total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
                max_width = max(line_widths) if line_widths else 0

                bg_left = x - padding
                bg_top = y - padding
                bg_right = x + max_width + padding
                bg_bottom = y + total_height + padding

                # Solid black background for high contrast
                draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=(0, 0, 0))

                # Draw each line with stroke for extra contrast
                current_y = y
                for i, text in enumerate(lines):
                    # Stroke
                    if self.stroke_width > 0:
                        for dx in [-self.stroke_width, 0, self.stroke_width]:
                            for dy in [-self.stroke_width, 0, self.stroke_width]:
                                if dx != 0 or dy != 0:
                                    draw.text((x + dx, current_y + dy), text, font=font, fill=self.stroke_color)
                    # Main text
                    draw.text((x, current_y), text, font=font, fill=self.text_color)

                    # Advance to next line
                    current_y += line_heights[i] + line_spacing
            
            # Convert back to numpy array
            return np.array(pil_image)
            
        except ImportError:
            # PIL not available, return original frame
            self.enable_overlay = False
            return frame
        except Exception as e:
            # If any error occurs, just return the original frame
            logger.warn(f"Error adding overlay to frame: {e}")
            return frame

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        
        # Increment episode counter and reset step counter
        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0
        
        self._capture_frame()

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()

        self.step_id += 1 # TODO: should this be incremented ever?
        self.current_step += 1
        
        # Accumulate reward for the selected env (default to first)
        if isinstance(rewards, np.ndarray):
            idx = int(self.record_env_idx or 0)
            idx = max(0, min(idx, rewards.shape[0] - 1))
            self.accumulated_reward += float(rewards[idx])
        else:
            # Non-vector case
            self.accumulated_reward += float(rewards)
        
        self._capture_frame()

        # Reset step counter if the selected environment is done
        if isinstance(dones, np.ndarray):
            idx = int(self.record_env_idx or 0)
            idx = max(0, min(idx, dones.shape[0] - 1))
            if bool(dones[idx]):
                self.current_step = 0
                self.current_episode += 1
                self.accumulated_reward = 0.0
        else:
            if bool(dones):
                self.current_step = 0
                self.current_episode += 1
                self.accumulated_reward = 0.0

        return obs, rewards, dones, infos

    def _capture_frame(self) -> None:
        if not self.recording: 
            return

        if self.video_length is not None and len(self.recorded_frames) >= self.video_length:
            return
        
        frame = None

        # Prefer capturing a single env image if requested and available
        if self.record_env_idx is not None:
            try:
                images = self.env.get_images()
                if isinstance(images, (list, tuple)) and len(images) > 0:
                    idx = int(self.record_env_idx)
                    # Clamp index to valid range
                    idx = max(0, min(idx, len(images) - 1))
                    img = images[idx]
                    if isinstance(img, np.ndarray):
                        frame = img
            except Exception:
                # Fall back to tiled render below
                frame = None

        # Fallback: use the VecEnv tiled render (all envs)
        if frame is None:
            frame = self.env.render()

        assert isinstance(frame, np.ndarray)
        frame_with_overlay = self._add_overlay_to_frame(frame)
        self.recorded_frames.append(frame_with_overlay)
        
    def close(self) -> None:
        """Closes the wrapper then the video recorder."""
        VecEnvWrapper.close(self)
        if self.recording:  # pragma: no cover
            self.stop_recording()

    def start_recording(self) -> None:
        self.recorded_frames = []
        self.recording = True

    def stop_recording(self) -> None:
        assert self.recording, "_stop_recording was called, but no recording was started"
        self.recording = False

    @contextmanager
    def recorder(self, video_path: str, record_video: bool = True):
        """Context manager for automatic recording start/stop.
        
        Usage:
            with recorder.recording_context():
                # Recording starts automatically
                for _ in range(100):
                    obs, rewards, dones, infos = env.step(actions)
                # Recording stops automatically when exiting the with block
        """
        # Normalize to string in case a PathLike was provided
        try:
            video_path = os.fspath(video_path)
        except Exception:
            # Fallback: ensure it's a string for downstream APIs
            video_path = str(video_path)

        if not record_video:
            yield self
            return

        video_root = os.path.dirname(video_path)
        os.makedirs(video_root, exist_ok=True)

        self.start_recording()
        try:
            yield self
        finally:
            # Always stop and save, even if an exception/early stop occurs
            try:
                self.stop_recording()
            except Exception:
                pass
            try:
                # Only save if we have at least one frame
                if len(self.recorded_frames) > 0:
                    self.save_recording(video_path)
            except Exception:
                # Avoid crashing the training due to video save failures
                pass

    def save_recording(self, video_path: str) -> None:
        assert len(self.recorded_frames) > 0, "No frames recorded to save."

        # Normalize potential PathLike to str for safety
        try:
            path_str = os.fspath(video_path)
        except Exception:
            path_str = str(video_path)

        assert path_str.endswith(".mp4"), "Video file must have .mp4 extension"

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
        clip.write_videofile(path_str, audio=False, logger=None)
        
        # Clear recorded frames after saving to prevent warning in __del__
        self.recorded_frames = []

    def __del__(self) -> None:
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:  # pragma: no cover
            logger.warn("Unable to save last video! Did you call close()?")
