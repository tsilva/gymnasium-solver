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

def build_env(env_id, n_envs=1, seed=None, norm_obs=False, norm_reward=False, vec_env_cls=None, reward_shaping=None, frame_stack=None):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
    
    if vec_env_cls == "SubProcVecEnv": vec_env_cls = SubprocVecEnv
    elif vec_env_cls == "DummyVecEnv": vec_env_cls = DummyVecEnv
    
    # Create env_fn with reward shaping for MountainCar
    def env_fn():
        if env_id == "MountainCar-v0":# and reward_shaping:
            from utils.wrappers import create_mountain_car_env
            return create_mountain_car_env(
                env_id=env_id,
                reward_shaping=True,
                normalize_obs=norm_obs,  # Use wrapper normalization instead of VecNormalize for MountainCar
                **reward_shaping if isinstance(reward_shaping, dict) else {}
            )
        else:
            import gymnasium as gym
            return gym.make(env_id)
    
    if env_id == "MountainCar-v0" and reward_shaping:
        # For MountainCar with reward shaping, use our custom env_fn
        env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
        # Only apply VecNormalize for reward normalization, not obs (handled by wrapper)
        if norm_reward: env = VecNormalize(env, norm_obs=False, norm_reward=norm_reward)
    else:
        # Standard environment creation
        env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
        if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    
    # Apply frame stacking if specified
    if frame_stack and frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    
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
