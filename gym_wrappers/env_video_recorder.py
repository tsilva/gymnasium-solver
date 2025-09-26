import os
from contextlib import contextmanager
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import logger

class EnvVideoRecorder(gym.Wrapper):
    """Record RGB frames from a single Gymnasium env to an MP4 file."""

    def __init__(
        self,
        env: gym.Env,
        video_length: Optional[int] = 200,
        enable_overlay: bool = True,
        font_size: int = 12,
        text_position: Tuple[int, int] = (6, 6),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        stroke_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_width: int = 2,
    ):
        super().__init__(env)

        render_mode = self.env.get_render_mode()
        assert render_mode == "rgb_array", (
            "EnvVideoRecorder requires the base env to be created with render_mode='rgb_array'"
        )

        self.frames_per_sec = self.env.get_render_fps()

        self.video_length = video_length
        self.recording = False
        self.recorded_frames: list[np.ndarray] = []

        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0

        self.enable_overlay = enable_overlay
        self.font_size = font_size
        self.text_position = tuple(text_position)
        self.text_color = tuple(text_color)
        self.stroke_color = tuple(stroke_color)
        self.stroke_width = stroke_width
        self._font = None

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

    def _capture_frame(self) -> None:
        if not self.recording:
            return
        if self.video_length is not None and len(self.recorded_frames) >= self.video_length:
            return

        frame = self.env.render()
        if not isinstance(frame, np.ndarray):
            try:
                frame = np.array(frame)
            except Exception as exc:  # pragma: no cover
                raise AssertionError("Env render returned unsupported frame type") from exc
        frame_with_overlay = self._add_overlay_to_frame(frame)
        self.recorded_frames.append(frame_with_overlay)

    def _on_episode_finished(self) -> None:
        self.current_step = 0
        self.current_episode += 1
        self.accumulated_reward = 0.0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.current_step = 0
        self.accumulated_reward = 0.0
        self._capture_frame()
        return result

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        self.current_step += 1
        self.accumulated_reward += float(reward)
        self._capture_frame()
        if done:
            self._on_episode_finished()

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def start_recording(self) -> None:
        self.recorded_frames = []
        self.current_episode = 0
        self.current_step = 0
        self.accumulated_reward = 0.0
        self.recording = True

    def stop_recording(self) -> None:
        assert self.recording, "stop_recording called without an active recording session"
        self.recording = False

    @contextmanager
    def recorder(self, video_path: str, record_video: bool = True):
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
