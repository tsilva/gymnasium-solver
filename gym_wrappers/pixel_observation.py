from typing import Any, Optional

import numpy as np
import gymnasium as gym


class PixelObservationWrapper(gym.ObservationWrapper):
    """Minimal pixel observation wrapper for Gymnasium.

    Replaces observations with RGB frames from env.render().
    Requires the base env to be created with render_mode='rgb_array'.
    """

    def __init__(self, env: gym.Env, *, pixels_only: bool = True, render_kwargs: Optional[dict] = None):
        super().__init__(env)
        self.pixels_only = pixels_only
        self.render_kwargs = render_kwargs or {}

        render_mode = getattr(self.env, "render_mode", None)
        if render_mode != "rgb_array":
            raise AttributeError(
                "env.render_mode must be 'rgb_array' when using PixelObservationWrapper. "
                "Create the env with render_mode='rgb_array'."
            )

        # Obtain a sample frame to set observation_space
        frame = self._render_frame()
        if frame is None:
            # Try resetting the environment once to allow rendering
            self.env.reset()
            frame = self._render_frame()
        if frame is None:
            raise RuntimeError("PixelObservationWrapper could not obtain a rendered frame from the environment.")

        frame = self._format_frame(frame)
        self._frame_shape = frame.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self._frame_shape, dtype=np.uint8)

    def _render_frame(self) -> Optional[np.ndarray]:
        frame = self.env.render(**self.render_kwargs)
        return frame

    @staticmethod
    def _format_frame(frame: Any) -> np.ndarray:
        arr = np.array(frame)
        # Ensure HWC uint8
        if arr.dtype != np.uint8:
            # If float in [0,1], scale to [0,255]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        # If CHW, transpose to HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def observation(self, observation):  # type: ignore[override]
        frame = self._render_frame()
        if frame is None:
            raise RuntimeError("PixelObservationWrapper failed to render a frame during observation()")
        return self._format_frame(frame)
