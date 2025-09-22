from contextlib import contextmanager
from typing import Optional

from gymnasium import error
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from gym_wrappers.env_video_recorder import EnvVideoRecorder
from gym_wrappers.utils import find_wrapper


# TODO: CLEANUP this file

class VecVideoRecorder(VecEnvWrapper):
    """Proxy VecEnv wrapper that delegates recording to EnvVideoRecorder instances."""

    def __init__(self, venv: VecEnv, record_env_idx: Optional[int] = 0):
        super().__init__(venv)
        self.record_env_idx = record_env_idx
        self.recording = False
        self.num_envs = venv.num_envs

    # ------------------------------------------------------------------ helpers

    def _resolve_env_index(self, idx_value: Optional[int], length: int) -> int:
        if idx_value is None:
            idx = 0
        else:
            try:
                idx = int(idx_value)
            except (TypeError, ValueError):
                idx = 0
        if length <= 0:
            return idx
        return max(0, min(idx, length - 1))

    def _unwrap_base_vec_env(self):
        base = self.venv
        while isinstance(base, VecEnvWrapper):
            base = base.venv
        return base

    def _get_env_recorder(self, env_idx: Optional[int] = None) -> EnvVideoRecorder:
        base = self._unwrap_base_vec_env()
        envs = getattr(base, "envs", None)
        if not isinstance(envs, list):
            raise error.Error(
                "VecVideoRecorder requires a DummyVecEnv-style base with direct env access"
            )
        idx = self._resolve_env_index(env_idx if env_idx is not None else self.record_env_idx, len(envs))
        target_env = envs[idx]
        recorder = find_wrapper(target_env, EnvVideoRecorder)
        if recorder is None:
            raise error.Error(
                "EnvVideoRecorder not found in the base env wrapper chain. Did you enable record_video?"
            )
        return recorder

    # ------------------------------------------------------------------ API

    def start_recording(self) -> None:
        recorder = self._get_env_recorder()
        recorder.start_recording()
        self.recording = True

    def stop_recording(self) -> None:
        recorder = self._get_env_recorder()
        recorder.stop_recording()
        self.recording = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    @contextmanager
    def recorder(self, video_path: str, record_video: bool = True):
        if not record_video:
            yield self
            return

        recorder = self._get_env_recorder()
        with recorder.recorder(video_path, record_video=True):
            self.recording = True
            try:
                yield self
            finally:
                self.recording = False

    def save_recording(self, video_path: str) -> None:
        recorder = self._get_env_recorder()
        recorder.save_recording(video_path)

    def close(self) -> None:
        if self.recording:
            try:
                self.stop_recording()
            except AssertionError:
                # Recorder may already be stopped by the underlying context manager
                pass
        super().close()
