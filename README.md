## gymnasium-solver 🤖🏋️

Fast, practical reinforcement learning on Gymnasium. Train PPO/REINFORCE agents with config-first workflows, vectorized environments, videos, a Gradio run inspector, and one-command publishing to the Hugging Face Hub.

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
- **Train** 🏃 (uses `--config_id "<env>:<variant>"`; defaults to `CartPole-v1:ppo` when omitted):
```bash
# Explicit config (recommended)
python train.py --config_id "CartPole-v1:ppo" -q
python train.py --config_id "CartPole-v1:reinforce" -q

# Uses train.py default when --config_id is omitted
python train.py -q
```
- **Play a trained policy** 🎮 (auto-loads best/last checkpoint from a run):
```bash
python play.py --run-id @latest-run --episodes 5
```
- **Inspect a run (UI)** 🔍:
```bash
python inspector.py --run-id @latest-run --port 7860 --host 127.0.0.1
```

### ⚙️ Configs (YAML)
Configs live in `config/environments/*.yaml`. New style puts base fields at the top and per-algorithm variants under their own key. Linear schedules like `lin_0.001` are parsed automatically. The loader selects an algo-specific config subclass based on `algo_id` (e.g., `PPOConfig`, `REINFORCEConfig`).

```yaml
# New per-file style
# Tip: omit project_id to default to the filename (e.g., CartPole-v1)
env_id: CartPole-v1
eval_episodes: 10

ppo:
  algo_id: ppo
  n_envs: 8
  max_timesteps: 1e5
  n_steps: 32
  batch_size: 256
  policy_lr: lin_0.001   # linear schedule from 0.001 → 0
  clip_range:   lin_0.2
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

Selection:
- Programmatic (Python): `load_config("CartPole-v1_ppo")` or `load_config("CartPole-v1", "ppo")`; omitting a variant defaults to the first defined (prefers `ppo`, then `reinforce`).
- CLI (train.py): pass `--config_id "<env>:<variant>"` (colon, not underscore), e.g., `--config_id "CartPole-v1:ppo"`.
The loader remains compatible with the legacy multi-block format for a transitional period. When `project_id` is omitted in environment YAMLs, it is inferred from the file name.

Key fields: `env_id`, `algo_id`, `n_envs`, `n_steps`, `batch_size`, `max_timesteps`, `policy` (`mlp|cnn`), `hidden_dims`, `obs_type` (`rgb|ram|objects` for ALE), `optimizer` (`adamw` default; supports `adam|adamw|sgd`).

Batch size can be either an absolute integer or a fraction in (0, 1]. If fractional, it is resolved as `batch_size = floor(n_envs * n_steps * fraction)`, with a minimum of 1.

- Advantage normalization: set `normalize_advantages` to `rollout` (normalize once per rollout, SB3-style), `batch` (normalize per mini-batch), or `off`.

REINFORCE options: set `returns_type` to control Monte Carlo returns used for scaling log-probs: `mc:rtg` (reward-to-go; default) or `mc:episode` (classic vanilla: make return constant across each episode segment).

VizDoom support: set `env_id` to `VizDoom-DeadlyCorridor-v0`. Requires `pip install vizdoom` and access to `deadly_corridor.cfg`/`deadly_corridor.wad` (auto-discovered from the installed package, or set `VIZDOOM_SCENARIOS_DIR` or `env_kwargs.config_path`).

### 🧰 Environment wrappers
Register-by-name wrappers via `EnvWrapperRegistry` (see `gym_wrappers/__init__.py`). Available IDs:
- `PixelObservationWrapper`
- `DiscreteEncoder` (encoding: 'array' | 'binary' | 'onehot')
- `PongV5_FeatureExtractor`, `PongV5_RewardShaper`
- `MountainCarV0_RewardShaper`, `CartPoleV1_RewardShaper`, `VizDoom_RewardShaper`

Use in YAML:
```yaml
env_wrappers:
  - { id: PixelObservationWrapper, pixels_only: true }
```

### 🎥 Runs, checkpoints, and videos
- 📁 Each training creates `runs/<id>/` with `config.json`, `checkpoints/*.ckpt`, `logs/`, and `videos/`
- 🔗 `runs/@latest-run` symlink points to the most recent run
- 🏷️ Best/last checkpoints: `best.ckpt`, `last.ckpt` (auto-detected by `play.py` and the inspector)
- 📈 Metrics: prints and logs `train/*` and `eval/*` including `ep_rew_mean` and running best as `ep_rew_best` (highlighted in blue; rules configurable in `config/metrics.yaml`). Metrics that fall outside configured bounds (`min`/`max` in `config/metrics.yaml`) are highlighted in yellow and emit a console warning. The console table also shows an inline ASCII sparkline (e.g., `█▇▇▆▅▄▃▂▁`) per numeric metric to visualize recent trends.

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
agents/            # PPO, REINFORCE
loggers/           # custom Lightning/CSV/console loggers
utils/             # config, env, models, rollouts, helpers
gym_wrappers/      # registry + wrappers (feature extractors, reward shaping, pixels)
trainer_callbacks/ # logging, early stopping, checkpointing, hyperparam sync, videos
config/            # environment YAML configs
runs/              # training outputs (checkpoints, videos, logs, config)
```

### 🧪 Tests
```bash
pytest -q
```

### 📄 License
MIT

### 🌀 W&B Sweeps
- Train under a sweep: set your sweep `program` to call `python train.py --config_id "<env>:<variant>"` (e.g., `CartPole-v1:ppo`).
- The script auto-detects W&B Agent via `WANDB_SWEEP_ID` and merges `wandb.config` into the main config before training. You can also force this behavior with `--wandb_sweep`.
- Parameter names map 1:1 to config fields (e.g., `n_envs`, `n_steps`, `batch_size`, `policy_lr`, `clip_range`, `gamma`, `gae_lambda`, `ent_coef`, `vf_coef`, `max_timesteps`, etc.). Linear schedules like `lin_0.001` are supported for sweep values.
- Fractional `batch_size` in (0, 1] is resolved as a fraction of `n_envs * n_steps` after overrides are applied.

Example sweep specs:
- Grid: `config/sweeps/cartpole_ppo_grid.yaml`
- Bayesian: `config/sweeps/cartpole_ppo_bayes.yaml`

Usage:
```bash
wandb sweep config/sweeps/cartpole_ppo_grid.yaml
wandb agent <entity>/<project>/<sweep_id>
```
