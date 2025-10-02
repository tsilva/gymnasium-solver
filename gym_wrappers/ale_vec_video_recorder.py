"""
Vector-level video recorder for ALE native vectorization.

Records observations from the first environment (index 0) in an ALE native
vectorized environment. Since ALE native vectorization returns preprocessed
observations (grayscale, frame-stacked), we capture these directly and
reconstruct viewable frames for video recording.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
from gymnasium import logger
from gymnasium.vector import VectorWrapper


class ALEVecVideoRecorder(VectorWrapper):
    """
    Records video from the first environment in an ALE native vectorized environment.

    Since ALE native vectorization preprocesses observations (grayscale, frame-stack),
    this recorder captures observations directly from step() and converts them back
    to RGB format suitable for video recording.

    The recorder tracks episode boundaries and episode statistics (steps, rewards)
    to overlay on the video frames.
    """

    def __init__(
        self,
        env,
        video_length: Optional[int] = 200,
        enable_overlay: bool = True,
        font_size: int = 12,
        text_position: tuple[int, int] = (6, 6),
        text_color: tuple[int, int, int] = (255, 255, 255),
        stroke_color: tuple[int, int, int] = (0, 0, 0),
        stroke_width: int = 2,
        target_env_idx: int = 0,
    ):
        super().__init__(env)

        # Video recording parameters
        self.video_length = video_length
        self.recording = False
        self.recorded_frames: list[np.ndarray] = []

        # Overlay parameters
        self.enable_overlay = enable_overlay
        self.font_size = font_size
        self.text_position = tuple(text_position)
        self.text_color = tuple(text_color)
        self.stroke_color = tuple(stroke_color)
        self.stroke_width = stroke_width
        self._font = None

        # Episode tracking for overlay
        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0

        # Track which environment to record (typically 0)
        self.target_env_idx = target_env_idx

        # Expected FPS for video encoding
        self.frames_per_sec = 60  # Default for Atari

        # Check if observations are grayscale (C, H, W) where C is frame stack
        obs_shape = self.single_observation_space.shape
        assert len(obs_shape) == 3, f"Expected (C, H, W) observation shape, got {obs_shape}"
        self.frame_stack, self.frame_height, self.frame_width = obs_shape
        self.is_grayscale = True  # ALE native vectorization uses grayscale by default

    def _get_font(self):
        if self._font is None:
            try:
                from PIL import ImageFont

                try:
                    self._font = ImageFont.truetype("DejaVuSansMono.ttf", size=self.font_size)
                except OSError:
                    try:
                        self._font = ImageFont.truetype("Arial.ttf", size=self.font_size)
                    except OSError:
                        self._font = ImageFont.load_default()
            except ImportError:
                self.enable_overlay = False
                self._font = None
        return self._font

    def _add_overlay_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add episode statistics overlay to a frame."""
        if not self.enable_overlay:
            return frame

        try:
            from PIL import Image, ImageDraw

            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)

            lines = [
                f"Episode: {self.current_episode + 1}",
                f"Step: {self.current_step}",
                f"Reward: {self.accumulated_reward:.2f}",
            ]

            font = self._get_font()
            if font is not None:
                x, y = self.text_position
                try:
                    bboxes = [draw.textbbox((0, 0), text, font=font) for text in lines]
                    line_heights = [(bbox[3] - bbox[1]) for bbox in bboxes]
                    line_widths = [(bbox[2] - bbox[0]) for bbox in bboxes]
                except (AttributeError, TypeError):
                    line_widths, line_heights = [], []
                    for text in lines:
                        width, height = font.getsize(text)
                        line_widths.append(width)
                        line_heights.append(height)

                line_spacing = max(1, int(self.font_size * 0.25))
                padding = 4
                total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
                max_width = max(line_widths) if line_widths else 0

                left = x - padding
                top = y - padding
                right = x + max_width + padding
                bottom = y + total_height + padding

                draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))

                current_y = y
                for idx, text in enumerate(lines):
                    if self.stroke_width > 0:
                        for dx in [-self.stroke_width, 0, self.stroke_width]:
                            for dy in [-self.stroke_width, 0, self.stroke_width]:
                                if dx != 0 or dy != 0:
                                    draw.text((x + dx, current_y + dy), text, font=font, fill=self.stroke_color)
                    draw.text((x, current_y), text, font=font, fill=self.text_color)
                    current_y += line_heights[idx] + line_spacing

            return np.array(pil_image)
        except ImportError:
            self.enable_overlay = False
            return frame

    def _obs_to_rgb_frame(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert a preprocessed observation to an RGB frame suitable for video.

        ALE native vectorization returns grayscale frame-stacked observations
        with shape (frame_stack, H, W). We take the most recent frame and
        convert to RGB by replicating the grayscale channel.

        Args:
            obs: Observation from target environment with shape (frame_stack, H, W)

        Returns:
            RGB frame with shape (H, W, 3) and dtype uint8
        """
        # Take the most recent frame (last in stack)
        assert obs.shape == (self.frame_stack, self.frame_height, self.frame_width), (
            f"Expected observation shape {(self.frame_stack, self.frame_height, self.frame_width)}, got {obs.shape}"
        )

        most_recent_frame = obs[-1]  # Shape: (H, W)

        # Convert grayscale to RGB by replicating channel
        rgb_frame = np.stack([most_recent_frame] * 3, axis=-1)  # Shape: (H, W, 3)

        assert rgb_frame.dtype == np.uint8, f"Expected uint8 frame, got {rgb_frame.dtype}"
        return rgb_frame

    def _capture_frame(self, obs: np.ndarray) -> None:
        """Capture a frame from the target environment's observation."""
        if not self.recording:
            return
        if self.video_length is not None and len(self.recorded_frames) >= self.video_length:
            return

        # Extract observation for target env
        target_obs = obs[self.target_env_idx]

        # Convert to RGB frame
        rgb_frame = self._obs_to_rgb_frame(target_obs)

        # Add overlay
        frame_with_overlay = self._add_overlay_to_frame(rgb_frame)

        self.recorded_frames.append(frame_with_overlay)

    def reset(self, **kwargs) -> Any:
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.accumulated_reward = 0.0
        self._capture_frame(obs)
        return obs, info

    def step(self, actions) -> Any:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # Track episode statistics for target env
        target_reward = float(rewards[self.target_env_idx])
        target_done = bool(terminations[self.target_env_idx] or truncations[self.target_env_idx])

        self.current_step += 1
        self.accumulated_reward += target_reward

        # Capture frame before potential episode boundary
        self._capture_frame(obs)

        # Handle episode completion
        if target_done:
            self.current_episode += 1
            self.current_step = 0
            self.accumulated_reward = 0.0

        return obs, rewards, terminations, truncations, infos

    def start_recording(self) -> None:
        """Start recording frames."""
        self.recorded_frames = []
        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0
        self.recording = True

    def stop_recording(self) -> None:
        """Stop recording frames."""
        assert self.recording, "stop_recording called without an active recording session"
        self.recording = False

    @contextmanager
    def recorder(self, video_path: str, record_video: bool = True):
        """
        Context manager for recording episodes to video.

        Args:
            video_path: Path to save the video file (must end in .mp4)
            record_video: Whether to actually record video (False for no-op)

        Yields:
            self
        """
        if not record_video:
            yield self
            return

        video_root = os.path.dirname(os.fspath(video_path))
        os.makedirs(video_root, exist_ok=True)

        self.start_recording()
        try:
            yield self
        finally:
            self.stop_recording()
            if len(self.recorded_frames) > 0:
                self.save_recording(video_path)

    def save_recording(self, video_path: str) -> None:
        """Save recorded frames to MP4 video file."""
        assert len(self.recorded_frames) > 0, "No frames recorded to save."
        path_str = os.fspath(video_path)
        assert path_str.endswith(".mp4"), "Video file must have .mp4 extension"

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
        clip.write_videofile(path_str, audio=False, logger=None)
        self.recorded_frames = []

    def close(self):
        if self.recording:
            self.stop_recording()
        return super().close()

    def __del__(self):
        frames = self.__dict__.get("recorded_frames", [])
        try:
            if isinstance(frames, list) and len(frames) > 0:
                logger.warn("Unable to save last video! Did you call close()?")
        except Exception:
            pass
