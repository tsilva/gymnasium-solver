"""Environment setup utilities."""

def build_env(env_id, n_envs=1, seed=None, norm_obs=False, norm_reward=False, vec_env_cls=None):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    if vec_env_cls == "SubProcVecEnv": vec_env_cls = SubprocVecEnv
    elif vec_env_cls == "DummyVecEnv": vec_env_cls = DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if norm_obs or norm_reward: env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env

def get_env_spec(env):
    return env.envs[0].spec

def get_env_reward_threshold(env):
    import gymnasium
    spec = get_env_spec(env)
    env_id = spec['env_id']
    spec = gymnasium.spec(env_id)
    return spec.reward_threshold


def _old_log_env_info(env) -> None:
    """Print key attributes of an environment or a vec-env.

    Handles:
      • Plain or wrapped Gym/Gymnasium envs
      • DummyVecEnv  (stores sub-envs locally)
      • SubprocVecEnv (sub-envs live in worker processes, accessed via RPC)

    Fields printed:
      Env ID, observation/action space, (reward range if available),
      max episode steps.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # ─────────────────────────────── helpers ──────────────────────────────────────
    def _compact(arr: np.ndarray) -> str:
        """Return a 1-D array as '[a, b, c]' with exactly one space after commas."""
        def fmt(x):
            if np.isposinf(x):
                return "inf"
            if np.isneginf(x):
                return "-inf"
            return f"{x:.3g}"          # 3 significant digits, no extra padding
        return "[" + ", ".join(fmt(v) for v in arr.ravel()) + "]"


    def _fmt_space(space) -> str:
        """Pretty-print Box / Discrete / etc. with compact low/high arrays."""
        if hasattr(space, "low") and hasattr(space, "high"):
            low  = _compact(space.low)
            high = _compact(space.high)
            return (f"{space.__class__.__name__}"
                    f"(low={low}, high={high}, shape={space.shape}, dtype={space.dtype})")
        return str(space)

    # 1) Detect vec-env type and pick a sub-env handle when possible
    if isinstance(env, DummyVecEnv):
        vec_kind, n = "DummyVecEnv", len(env.envs)
        base_env    = env.envs[0]                  # local object
    elif isinstance(env, SubprocVecEnv):
        vec_kind, n = "SubprocVecEnv", env.num_envs
        base_env    = None                         # must query via RPC
    else:
        vec_kind, n = None, 1                      # single or wrapped env
        base_env    = env

    # 2) Safe remote attribute fetcher for SubprocVecEnv
    def remote_attr(name):
        if not isinstance(env, SubprocVecEnv):
            return None
        try:
            return env.get_attr(name, indices=0)[0]
        except Exception:
            return None                            # Attribute missing ➞ None

    # 3) Gather ID, max-steps, reward range
    if base_env is not None:                       # DummyVecEnv or plain env
        spec       = getattr(base_env, "spec", None)
        env_id     = getattr(spec, "id", None) or getattr(base_env, "id", "Unknown")
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or getattr(base_env, "_max_episode_steps", "Unknown")
        reward_rng = getattr(base_env, "reward_range", None)  # Gym only
    else:                                          # SubprocVecEnv
        spec       = remote_attr("spec")
        env_id     = getattr(spec, "id", None) or remote_attr("id") or "Unknown"
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or remote_attr("_max_episode_steps") or "Unknown"
        reward_rng = None                          # don’t fetch reward_range remotely

    # 4) Observation / action spaces are exposed on the vec-env itself
    obs_space = env.observation_space
    act_space = env.action_space

    # 5) Print results
    header = f"Environment Info ({vec_kind} with {n} envs)" if vec_kind else "Environment Info"
    print(header)
    print(f"  Env ID: {env_id}")
    print(f"  Observation space: {_fmt_space(obs_space)}")
    print(f"  Action space: {_fmt_space(act_space)}")
    if reward_rng is not None:
        print(f"  Reward range: {reward_rng}")
    print(f"  Max episode steps: {max_steps}")


import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Dict, Any


# ───────────────────────── helpers ─────────────────────────────────────────────
def _compact(arr: np.ndarray) -> str:
    """Return a 1-D array as '[a, b, c]' with exactly one space after commas."""
    def fmt(x):
        if np.isposinf(x):
            return "inf"
        if np.isneginf(x):
            return "-inf"
        return f"{x:.3g}"          # 3 significant digits, no extra padding
    return "[" + ", ".join(fmt(v) for v in arr.ravel()) + "]"


def _fmt_space(space) -> str:
    """Pretty-print Box / Discrete / etc. with compact low/high arrays."""
    if hasattr(space, "low") and hasattr(space, "high"):
        low  = _compact(space.low)
        high = _compact(space.high)
        return (f"{space.__class__.__name__}"
                f"(low={low}, high={high}, shape={space.shape}, dtype={space.dtype})")
    return str(space)


# ──────────────────────── data-gathering core ────────────────────────────────
def get_env_spec(env) -> Dict[str, Any]:
    """
    Inspect an environment (plain, DummyVecEnv, or SubprocVecEnv) and return the
    key characteristics in a dictionary.  No printing here.
    """
    # 1) Detect vec-env type and pick a sub-env handle when possible
    if isinstance(env, DummyVecEnv):
        vec_kind, n = "DummyVecEnv", len(env.envs)
        base_env    = env.envs[0]                  # local object
    elif isinstance(env, SubprocVecEnv):
        vec_kind, n = "SubprocVecEnv", env.num_envs
        base_env    = None                         # must query via RPC
    else:
        vec_kind, n = None, 1                      # single or wrapped env
        base_env    = env

    # 2) Safe remote attribute fetcher for SubprocVecEnv
    def remote_attr(name):
        if not isinstance(env, SubprocVecEnv):
            return None
        try:
            return env.get_attr(name, indices=0)[0]
        except Exception:
            return None                            # Attribute missing → None

    # 3) Gather ID, max-steps, reward range
    if base_env is not None:                       # DummyVecEnv or plain env
        spec       = getattr(base_env, "spec", None)
        env_id     = getattr(spec, "id", None) or getattr(base_env, "id", "Unknown")
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or getattr(base_env, "_max_episode_steps", "Unknown")
        reward_rng = getattr(base_env, "reward_range", None)  # Gym only
    else:                                          # SubprocVecEnv
        spec       = remote_attr("spec")
        env_id     = getattr(spec, "id", None) or remote_attr("id") or "Unknown"
        max_steps  = getattr(spec, "max_episode_steps", None) \
                     or remote_attr("_max_episode_steps") or "Unknown"
        reward_rng = None                          # don’t fetch reward_range remotely

    # 4) Observation / action spaces are exposed on the vec-env itself
    obs_space = env.observation_space
    act_space = env.action_space

    # 5) Assemble dictionary
    return {
        "vec_kind"      : vec_kind,
        "num_envs"      : n,
        "env_id"        : env_id,
        "obs_space_str" : _fmt_space(obs_space),
        "act_space_str" : _fmt_space(act_space),
        "input_dim"    : int(env.observation_space.shape[0]),
        "output_dim"   : int(env.action_space.n),
        "reward_range"  : reward_rng,
        "max_steps"     : max_steps,
    }


# ─────────────────────── public logging helper ───────────────────────────────
def log_env_info(env) -> None:
    """Pretty-print the information returned by `gather_env_info`."""
    info = get_env_spec(env)

    header = (f"Environment Info ({info['vec_kind']} with {info['num_envs']} envs)"
              if info["vec_kind"] else "Environment Info")
    print(header)
    print(f"  Env ID: {info['env_id']}")
    print(f"  Observation space: {info['obs_space_str']}")
    print(f"  Action space: {info['act_space_str']}")
    if info["reward_range"] is not None:
        print(f"  Reward range: {info['reward_range']}")
    print(f"  Max episode steps: {info['max_steps']}")
    
import numpy as np
from typing import Iterable, Tuple
import os
from pathlib import Path
from PIL import ImageFont
from IPython.display import HTML
import tempfile
import subprocess
import uuid
import shutil
from collections.abc import Sequence

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
    dones = trajectories[3]
    frames = trajectories[8]
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
