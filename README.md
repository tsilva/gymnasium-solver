## gymnasium-solver 🤖🏋️

Fast, practical reinforcement learning on Gymnasium. Train PPO/REINFORCE/QLearning agents with config-first workflows, vectorized environments, videos, a Gradio run inspector, and one-command publishing to the Hugging Face Hub.

### ⚠️ Warning
This project is currently for self-education purposes only. I'm doing a lot of vibe coding: I quickly vibe-code features, then review the code and chisel out the AI slop. At any point the codebase may be ugly and buggy. Please don't assume it's a "real" project until I make the first release. From that point on, I'll start working with branches and a more stable workflow.

### ✨ Highlights
- **Algorithms** 🧠: PPO, REINFORCE, Q-Learning
- **Config-first** ⚙️: concise YAML configs with inheritance and linear schedules (e.g., `lin_0.001`)
- **Vectorized envs** ⚡: Dummy/Subproc, frame stacking, obs/reward normalization
- **Atari-ready** 🕹️: ALE with `obs_type` rgb/ram/objects (via [Gymnasium](https://gymnasium.farama.org) and [OCAtari](https://github.com/Kautenja/oc-atari))
- **Retro-ready** 🎮: Classic console games via [stable-retro](https://github.com/Farama-Foundation/stable-retro) (e.g., `Retro/SuperMarioBros-Nes`)
- **Wrappers registry** 🧰: plug-in env wrappers by name
- **Great UX** ✨: curated `runs/` folders, auto `@latest-run` link, video capture
- **Inspector UI** 🔎: step-by-step episode browser (Gradio)
- **Hub publishing** 📤: push run artifacts and preview video to [Hugging Face Hub](https://huggingface.co)

### 📦 Install
- Using uv (recommended):
```bash
pipx install uv  # or: pip install uv
uv sync
```
- Or with pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 🚀 Quickstart
- **Train** 🏃 (uses YAML config IDs from `config/environments/*.yaml`):
```bash
python train.py --config CartPole-v1_ppo -q
```
- **Play a trained policy** 🎮 (auto-loads best/last checkpoint from a run):
```bash
python play.py --run-id @latest-run --episodes 5
```
- **Inspect a run (UI)** 🔍:
```bash
python inspect.py --run-id @latest-run --port 7860 --host 127.0.0.1
```

### ⚙️ Configs (YAML)
Configs live in `config/environments/*.yaml`. New style puts base fields at the top and per-algorithm variants under their own key. Linear schedules like `lin_0.001` are parsed automatically.

```yaml
# New per-file style
# Tip: omit project_id to default to the filename (e.g., CartPole-v1)
env_id: CartPole-v1
eval_episodes: 10

ppo:
  algo_id: ppo
  n_envs: 8
  n_timesteps: 1e5
  n_steps: 32
  batch_size: 256
  learning_rate: lin_0.001   # linear schedule from 0.001 → 0
  clip_range:   lin_0.2
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

You still select a config by ID, e.g. `CartPole-v1_ppo`. The loader also remains compatible with the legacy multi-block format for a transitional period. When `project_id` is omitted in environment YAMLs, it is inferred from the file name.

Key fields: `env_id`, `algo_id`, `n_envs`, `n_steps`, `batch_size`, `n_timesteps`, `policy` (`mlp|cnn`), `hidden_dims`, `obs_type` (`rgb|ram|objects` for ALE).

VizDoom support: set `env_id` to `VizDoom-DeadlyCorridor-v0`. Requires `pip install vizdoom` and access to `deadly_corridor.cfg`/`deadly_corridor.wad` (auto-discovered from the installed package, or set `VIZDOOM_SCENARIOS_DIR` or `env_kwargs.config_path`).

### 🧰 Environment wrappers
Register-by-name wrappers via `EnvWrapperRegistry` (see `gym_wrappers/__init__.py`). Available IDs:
- `PixelObservationWrapper`
- `DiscreteToBinary`
- `PongV5_FeatureExtractor`, `PongV5_RewardShaper`
- `MountainCarV0_RewardShaper`, `CartPoleV1_RewardShaper`

Use in YAML:
```yaml
env_wrappers:
  - { id: PixelObservationWrapper, pixels_only: true }
```

### 🎥 Runs, checkpoints, and videos
- 📁 Each training creates `runs/<id>/` with `config.json`, `checkpoints/*.ckpt`, `logs/`, and `videos/`
- 🔗 `runs/@latest-run` symlink points to the most recent run
- 🏷️ Best/last checkpoints: `best.ckpt`, `last.ckpt` (auto-detected by `play.py` and the inspector)

### 📤 Publish to Hugging Face Hub
Authenticate once (`huggingface-cli login`) or set `HF_TOKEN`, then:
```bash
python publish.py                 # publish latest run
python publish.py --run-id <ID>   # publish a specific run
python publish.py --repo user/repo --private
```
Uploads run artifacts under `artifacts/` and attaches a preview video when found.

### 🗂️ Project layout
```
agents/           # PPO, REINFORCE, QLearning
utils/            # config, env, logging, models, rollouts, etc.
gym_wrappers/     # registry + wrappers (feature extractors, reward shaping, pixels)
config/           # environment YAML configs
runs/             # training outputs (checkpoints, videos, logs, config)
```

### 🧪 Tests
```bash
pytest -q
```

### 📄 License
MIT
