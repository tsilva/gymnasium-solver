### 
# DISCLAIMER: originally from stable_baselines3/common/vec_env/vec_video_recorder.py
###

import os
import os.path
from typing import Optional, Tuple

import numpy as np
from gymnasium import error, logger

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
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
    """

    def __init__(
        self,
        venv: VecEnv,
        video_length: Optional[int] = 200,
        # Text overlay options
        enable_overlay: bool = True,
        font_size: int = 24,
        text_position: Tuple[int, int] = (10, 10),
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

        self.recording = False
        self.recorded_frames: list[np.ndarray] = []
        
        # Episode and step tracking for overlay
        self.current_episode = 0
        self.current_step = 0
        
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
        """Add episode and step overlay to the frame."""
        if not self.enable_overlay:
            return frame
            
        try:
            from PIL import Image, ImageDraw
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            
            # Create overlay text
            text = f"Episode: {self.current_episode + 1}  Step: {self.current_step + 1}"
            
            # Get font
            font = self._get_font()
            
            if font is not None:
                # Draw text with stroke (outline)
                x, y = self.text_position
                
                # Draw stroke by drawing text in multiple positions
                for dx in [-self.stroke_width, 0, self.stroke_width]:
                    for dy in [-self.stroke_width, 0, self.stroke_width]:
                        if dx != 0 or dy != 0:  # Don't draw at center position yet
                            draw.text(
                                (x + dx, y + dy),
                                text,
                                font=font,
                                fill=self.stroke_color
                            )
                
                # Draw main text
                draw.text(
                    self.text_position,
                    text,
                    font=font,
                    fill=self.text_color
                )
            
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
        self.current_episode += 1
        self.current_step = 0
        
        if self.recording: self._capture_frame()

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()

        self.step_id += 1 # TODO: should this be incremented ever?
        self.current_step += 1
        
        if self.recording:
            self._capture_frame()

        # Reset step counter if any environment is done
        if np.any(dones):
            self.current_step = 0
            self.current_episode += 1 # TODO: doesnt work for multiple envs

        return obs, rewards, dones, infos

    def _capture_frame(self) -> None:
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        if self.video_length is not None and len(self.recorded_frames) >= self.video_length:
            return
        
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

    def save_recording(self, video_path: str) -> None:
        assert len(self.recorded_frames) > 0, "No frames recorded to save."

        assert video_path.endswith(".mp4"), "Video file must have .mp4 extension"

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
        clip.write_videofile(video_path, audio=False, logger=None)

    def __del__(self) -> None:
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:  # pragma: no cover
            logger.warn("Unable to save last video! Did you call close()?")
