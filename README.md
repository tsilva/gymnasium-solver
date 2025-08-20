## gymnasium-solver

Fast, practical reinforcement learning on Gymnasium. Train PPO/REINFORCE/QLearning agents with config-first workflows, vectorized environments, videos, a Gradio run inspector, and one-command publishing to the Hugging Face Hub.

### Highlights
- **Algorithms**: PPO, REINFORCE, Q-Learning
- **Config-first**: concise YAML configs with inheritance and linear schedules (e.g., `lin_0.001`)
- **Vectorized envs**: Dummy/Subproc, frame stacking, obs/reward normalization
- **Atari-ready**: ALE with `obs_type` rgb/ram/objects (via [Gymnasium](https://gymnasium.farama.org) and [OCAtari](https://github.com/Kautenja/oc-atari))
- **Wrappers registry**: plug-in env wrappers by name
- **Great UX**: curated `runs/` folders, auto `latest-run` link, video capture
- **Inspector UI**: step-by-step episode browser (Gradio)
- **Hub publishing**: push run artifacts and preview video to [Hugging Face Hub](https://huggingface.co)

### Install
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

### Quickstart
- **Train** (uses YAML config IDs from `config/environments/*.yaml`):
```bash
python train.py --config CartPole-v1_ppo -q
```
- **Play a trained policy** (auto-loads best/last checkpoint from a run):
```bash
python play.py --run-id latest-run --episodes 5
```
- **Inspect a run (UI)**:
```bash
python inspect.py --run-id latest-run --port 7860 --host 127.0.0.1
```

### Configs (YAML)
Configs live in `config/environments/*.yaml`. They support inheritance and linear schedules.

```yaml
CartPole-v1_ppo:
  inherits: __CartPole-v1
  algo_id: ppo
  n_envs: 8
  n_timesteps: 1e5
  n_steps: 32
  batch_size: 256
  learning_rate: lin_0.001   # linear schedule from 0.001 → 0
  clip_range:   lin_0.2
  # optional wrappers
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

Key fields: `env_id`, `algo_id`, `n_envs`, `n_steps`, `batch_size`, `n_timesteps`, `policy` (`mlp|cnn`), `hidden_dims`, `obs_type` (`rgb|ram|objects` for ALE).

### Environment wrappers
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

### Runs, checkpoints, and videos
- Each training creates `runs/<id>/` with `config.json`, `checkpoints/*.ckpt`, `logs/`, and `videos/`
- `runs/latest-run` symlink points to the most recent run
- Best/last checkpoints: `best.ckpt`, `last.ckpt` (auto-detected by `play.py` and the inspector)

### Publish to Hugging Face Hub
Authenticate once (`huggingface-cli login`) or set `HF_TOKEN`, then:
```bash
python publish.py                 # publish latest run
python publish.py --run-id <ID>   # publish a specific run
python publish.py --repo user/repo --private
```
Uploads run artifacts under `artifacts/` and attaches a preview video when found.

### Project layout
```
agents/           # PPO, REINFORCE, QLearning
utils/            # config, env, logging, models, rollouts, etc.
gym_wrappers/     # registry + wrappers (feature extractors, reward shaping, pixels)
config/           # environment YAML configs
runs/             # training outputs (checkpoints, videos, logs, config)
```

### Tests
```bash
pytest -q
```

### License
MIT