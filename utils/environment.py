"""Environment setup utilities."""

import numpy as np
from typing import Iterable, Tuple, Dict, Any
import os
from pathlib import Path
from PIL import ImageFont
from IPython.display import HTML
import tempfile
import subprocess
import uuid
import gymnasium
import shutil
from collections.abc import Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn

class VecNormalizeStatic(VecEnvWrapper):
    """
    A simple VecEnv wrapper that normalizes observations to [0, 1]
    based on the static observation space bounds (Box low/high).

    - Does NOT normalize rewards.
    - Does NOT track running stats.
    """

    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        assert isinstance(venv.observation_space, spaces.Box), "Only supports Box observation spaces."
        self.low = venv.observation_space.low.astype(np.float32)
        self.high = venv.observation_space.high.astype(np.float32)
        self.scale = self.high - self.low
        # Update observation space to reflect normalized range
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=venv.observation_space.shape,
            dtype=np.float32,
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) - self.low) / (self.scale + 1e-8)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return self._normalize_obs(obs)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._normalize_obs(obs), rewards, dones, infos

"""Environment wrappers for reward shaping and other modifications."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class MountainCarRewardShaping(gym.Wrapper):
    """
    Reward shaping wrapper for MountainCar-v0 to help with convergence.
    
    The original MountainCar has sparse rewards:
    - +100 for reaching the goal (position >= 0.5)
    - -1 for each step otherwise
    
    This wrapper adds dense shaping rewards based on:
    1. Position progress toward goal (rightward movement)
    2. Velocity in the right direction 
    3. Height gained on the mountain
    
    The shaping rewards are designed to be potential-based to maintain optimality.
    """
    
    def __init__(self, env, position_reward_scale=100.0, velocity_reward_scale=10.0, height_reward_scale=50.0):
        super().__init__(env)
        self.position_reward_scale = position_reward_scale
        self.velocity_reward_scale = velocity_reward_scale  
        self.height_reward_scale = height_reward_scale
        
        # MountainCar bounds
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        
        # Previous state for computing deltas
        self.prev_position = None
        self.prev_velocity = None
        self.prev_height = None
        
    def _get_height(self, position):
        """Calculate height on the mountain based on position."""
        # Mountain Car height function: sin(3 * position)
        return np.sin(3 * position)
    
    def _get_position_potential(self, position):
        """Position-based potential: closer to goal = higher potential."""
        # Normalize position to [0, 1] where 1 is at goal
        normalized_pos = (position - self.min_position) / (self.goal_position - self.min_position)
        return normalized_pos
    
    def _get_velocity_potential(self, velocity):
        """Velocity-based potential: positive velocity = higher potential.""" 
        # Normalize velocity to [0, 1] where 1 is max positive velocity
        normalized_vel = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity)
        return normalized_vel
    
    def _get_height_potential(self, height):
        """Height-based potential: higher on mountain = higher potential."""
        # Height ranges from -1 to 1, normalize to [0, 1]
        return (height + 1) / 2
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        position, velocity = obs
        
        # Initialize previous state
        self.prev_position = position
        self.prev_velocity = velocity  
        self.prev_height = self._get_height(position)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs
        height = self._get_height(position)
        
        # Calculate potential-based shaping rewards
        shaping_reward = 0.0
        
        if self.prev_position is not None:
            # Position progress reward (potential difference)
            curr_pos_potential = self._get_position_potential(position)
            prev_pos_potential = self._get_position_potential(self.prev_position)
            position_shaping = self.position_reward_scale * (curr_pos_potential - prev_pos_potential)
            
            # Velocity shaping reward (encourage positive velocity)
            curr_vel_potential = self._get_velocity_potential(velocity)
            prev_vel_potential = self._get_velocity_potential(self.prev_velocity)
            velocity_shaping = self.velocity_reward_scale * (curr_vel_potential - prev_vel_potential)
            
            # Height progress reward (potential difference)
            curr_height_potential = self._get_height_potential(height)
            prev_height_potential = self._get_height_potential(self.prev_height)
            height_shaping = self.height_reward_scale * (curr_height_potential - prev_height_potential)
            
            shaping_reward = position_shaping + velocity_shaping + height_shaping
            
            # Add debug info
            info['shaping_reward'] = shaping_reward
            info['position_shaping'] = position_shaping
            info['velocity_shaping'] = velocity_shaping
            info['height_shaping'] = height_shaping
        
        # Update previous state
        self.prev_position = position
        self.prev_velocity = velocity
        self.prev_height = height
        
        # Add shaping reward to original reward
        shaped_reward = reward + shaping_reward
        
        return obs, shaped_reward, terminated, truncated, info
    
# TODO: softcode this
def is_atari_env_id(env_id: str) -> bool:
    return env_id.startswith("ALE/") or env_id in ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]

def build_env(
    env_id, 
    n_envs=1, 
    seed=None, 
    norm_obs=False, 
    norm_reward=False, 
    vec_env_cls=None, 
    reward_shaping=None, 
    frame_stack=None, 
    obs_type=None,
    record_video=False, 
    record_video_kwargs={}
):
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
    
    if vec_env_cls == "SubProcVecEnv": vec_env_cls = SubprocVecEnv
    elif vec_env_cls == "DummyVecEnv": vec_env_cls = DummyVecEnv
    
    render_mode = "rgb_array" if record_video else None

    # Create env_fn with reward shaping for MountainCar and obs_type for Atari
    def env_fn():
        if is_atari_env_id(env_id):
            import ale_py
            gymnasium.register_envs(ale_py) # TODO: do this only once
            env = gym.make(env_id, obs_type=obs_type, render_mode=render_mode)
        else:
            # TODO: softcode wrappers
            env = gym.make(env_id, render_mode=render_mode)
            if env_id == "MountainCar-v0" and reward_shaping: env = MountainCarRewardShaping(env)
            return env

    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    # TODO: softcode to use this optionally (we want this in ram envs)
    if norm_obs == "static": env = VecNormalizeStatic(env)
    elif norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    
    # Apply frame stacking if specified
    if frame_stack and frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    
    if record_video:
        from wrappers.vec_video_recorder import VecVideoRecorder
        env = VecVideoRecorder(
            env,
            record_video_kwargs
        )

    return env

def get_env_spec(env: gymnasium.Env | str) -> Dict[str, Any]:
    if type(env) is str: return gymnasium.spec(env)
    return env.envs[0].spec

def get_env_reward_threshold(env):
    import gymnasium
    spec = get_env_spec(env)
    env_id = spec.id#['id']
    spec = gymnasium.spec(env_id)
    return spec.reward_threshold

def render_episode_frames(
    frames: Iterable[np.ndarray] | Iterable[Iterable[np.ndarray]],
    *,
    fps: int = 30,
    out_path: str | os.PathLike | None = None,
    out_dir: str | os.PathLike | None = None,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
    # ---------- label styling ----------------------------------------
    font: ImageFont.ImageFont | None = None,
    text_xy: Tuple[int, int] = (5, 5),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 1,
    # ---------- outer grid size --------------------------------------
    grid: Tuple[int, int] = (1, 1),
    # ---------- embed options ----------------------------------------
    width: int = 640,
) -> HTML:
    """
    Encode *frames* (flat list) or *episodes* (list of lists) into an MP4
    and return an embeddable HTML snippet.

    Notes
    -----
    * If ``out_dir`` is given and **lies outside** the current notebook
      directory, the video is silently **copied back** next to the
      notebook so the browser can reach it.
    """

    import itertools
    import numpy as np
    from IPython.display import HTML
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v3 as iio

    # ---------------- font --------------------------------------------------
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size=24)
        except OSError:
            font = ImageFont.load_default()

    # ---------------- flat vs. nested detection ----------------------------
    fr_iter = iter(frames)
    try:
        first_item = next(fr_iter)
    except StopIteration as e:
        raise ValueError("No frames provided.") from e
    fr_iter = itertools.chain([first_item], fr_iter)

    nested = isinstance(first_item, (Sequence, Iterable)) and not (
        isinstance(first_item, np.ndarray) and first_item.ndim == 3
    )

    episodes = (
        [list(ep) for ep in fr_iter] if nested else [list(fr_iter)]
    )
    if not episodes or not episodes[0]:
        raise ValueError("No frames provided.")

    # ---------------- validate frames --------------------------------------
    ref_H, ref_W, _ = episodes[0][0].shape
    if not (
        isinstance(episodes[0][0], np.ndarray)
        and episodes[0][0].dtype == np.uint8
        and episodes[0][0].ndim == 3
    ):
        raise ValueError("Frames must be uint8 RGB arrays of shape (H,W,3).")

    for ep in episodes:
        for fr in ep:
            if fr.shape != (ref_H, ref_W, 3):
                raise ValueError("All frames must share the same (H,W,3).")

    # ---------------- outer grid geometry ----------------------------------
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError("grid must contain positive integers.")
    n_cells = rows * cols

    # partition episodes -> cells, round-robin
    cell_to_eps = {c: [] for c in range(n_cells)}
    for ep_idx, cell in enumerate(
        itertools.islice(itertools.cycle(range(n_cells)), len(episodes))
    ):
        cell_to_eps[cell].append(ep_idx)

    # build per-cell flat playback sequence [(ep, step, frame), ...]
    cell_seq: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for cell, eps in cell_to_eps.items():
        seq = []
        for ep_idx in eps:
            for st_idx, fr in enumerate(episodes[ep_idx]):
                seq.append((ep_idx, st_idx, fr))
        if not seq:  # empty slot → black frame
            black = np.zeros_like(episodes[0][0])
            seq = [(-1, -1, black)]
        cell_seq[cell] = seq

    max_len = max(len(s) for s in cell_seq.values())

    # ---------------- helper: stamp label ----------------------------------
    def stamp(img: Image.Image, label: str):
        draw = ImageDraw.Draw(img, "RGB")
        draw.text(
            text_xy,
            label,
            font=font,
            fill=text_color,
            stroke_fill=stroke_color,
            stroke_width=stroke_width,
        )

    # ---------------- temp dir for PNGs ------------------------------------
    with tempfile.TemporaryDirectory() as tmp_root:
        png_dir = Path(tmp_root) / "frames"
        png_dir.mkdir()

        frame_id = 0
        for t in range(max_len):
            canvas = np.zeros((rows * ref_H, cols * ref_W, 3), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    cell = r * cols + c
                    seq = cell_seq[cell]
                    ep_idx, st_idx, frame = (
                        seq[t] if t < len(seq) else seq[-1]
                    )
                    img = Image.fromarray(frame)
                    if ep_idx >= 0:
                        stamp(img, f"Episode: {ep_idx + 1}  Step: {st_idx + 1}")
                    y0, x0 = r * ref_H, c * ref_W
                    canvas[y0 : y0 + ref_H, x0 : x0 + ref_W] = np.asarray(img)

            iio.imwrite(
                png_dir / f"frame_{frame_id:06d}.png",
                canvas,
                plugin="pillow",
            )
            frame_id += 1

        # ---------------- encode with FFmpeg -------------------------------
        if out_path is None:
            fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            out_path_ = Path(tmp_name)
        else:
            out_path_ = Path(out_path)

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(png_dir / "frame_%06d.png"),
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(out_path_),
        ]
        subprocess.run(cmd, check=True)

    # ---------------- move / copy video for the notebook -------------------
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)

    dest = out_dir / f"video_{uuid.uuid4().hex}.mp4"
    if out_path_ != dest:
        shutil.move(out_path_, dest)

    # Ensure the browser can reach the file
    nb_root = Path.cwd().resolve()
    try:
        # Jupyter can serve anything under the notebook root
        src_path = dest.resolve().relative_to(nb_root).as_posix()
    except ValueError:
        # File lies outside → copy it next to the notebook
        safe_dest = nb_root / dest.name
        shutil.copy(dest, safe_dest)
        src_path = safe_dest.name

    # ---------------- build HTML snippet -----------------------------------
    vid_id = f"vid_{uuid.uuid4().hex}"
    html = f"""
<div style="max-width:{width}px">
    <video id="{vid_id}" src="{src_path}" width="100%" controls playsinline
            style="background:#000"></video>

    <div style="font-size:0.9em;margin-top:4px">
    <a href="{src_path}" target="_blank">🔗 open in new tab (fullscreen ✓)</a>
    &nbsp;•&nbsp; double-click video to toggle fullscreen
    </div>
</div>

<script>
const v = document.getElementById('{vid_id}');
v.addEventListener('dblclick', () => {{
    if (!document.fullscreenElement) {{
        (v.requestFullscreen || v.webkitRequestFullscreen ||
            v.msRequestFullscreen).call(v);
    }} else {{
        (document.exitFullscreen || document.webkitExitFullscreen ||
            document.msExitFullscreen).call(document);
    }}
}});
</script>
""".strip()
    
    return HTML(html)

def group_frames_by_episodes(trajectories):
    dones = trajectories.dones
    frames = trajectories[8]  # frames is still accessed by index since it's not part of the named tuple
    assert len(dones) == len(frames), "Dones and frames must have the same length"

    episodes = []
    current_episode = []

    for frame, done in zip(frames, dones):
        current_episode.append(frame)
        if not done.item(): continue
        episodes.append(current_episode)
        current_episode = []

    # Optionally handle case where last episode does not end with a `done`
    if current_episode:
        episodes.append(current_episode)

    return episodes
