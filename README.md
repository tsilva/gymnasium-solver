# gymnasium-solver

Fast, config-first reinforcement learning framework built on PyTorch Lightning and Gymnasium. Train PPO and REINFORCE agents with vectorized environments, video capture, and seamless W&B/Hugging Face Hub integration.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Warning

**This is a self-education project undergoing rapid development ("vibe coding").** Expect instability and breaking changes until the first official release. The codebase may be ugly or buggy at any point. Do not use in production.

## Features

- **Algorithms**: PPO, REINFORCE with policy-only or actor-critic architectures
- **Config-first**: YAML configs with inheritance, variants, and dict-based schedules (`{start: 0.001, end: 0.0}`)
- **Vectorized environments**: Sync/async modes, frame stacking, observation/reward normalization
- **Atari support**: ALE with `obs_type` rgb/ram/objects via [Gymnasium](https://gymnasium.farama.org) and [OCAtari](https://github.com/Kautenja/oc-atari)
- **Retro support**: Classic console games via [stable-retro](https://github.com/Farama-Foundation/stable-retro) (optional; broken on M1 Mac)
- **VizDoom support**: First-person shooter environments
- **Wrapper registry**: Plug-in environment wrappers by name
- **Run management**: Clean `runs/` structure with `@last` symlink, automatic best/last checkpoints
- **Video capture**: Automatic episode recording during evaluation
- **Inspector UI**: Gradio-based step-by-step episode browser with frame visualization
- **W&B integration**: Automatic dashboard creation, metrics tracking, video uploads
- **W&B Sweeps**: Local and distributed (Modal AI) hyperparameter optimization
- **Hugging Face Hub**: One-command publishing of trained models
- **Hyperparameter schedules**: Linear interpolation for learning rates, clip ranges, entropy coefficients
- **CLI overrides**: Override config values without editing YAML files

## Installation

### Using uv (recommended)

```bash
pipx install uv  # or: pip install uv
uv sync
```

### Using pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quick Start

### Training

Train an agent using config specs in `<env>:<variant>` format:

```bash
# Train PPO on CartPole
python train.py CartPole-v1:ppo -q

# Train REINFORCE
python train.py CartPole-v1:reinforce -q

# Use explicit flag
python train.py --config_id "CartPole-v1:ppo" -q

# Defaults to Bandit-v0:ppo when no config specified
python train.py -q

# Override max timesteps from CLI
python train.py CartPole-v1:ppo --max-timesteps 5000

# List available environments
python train.py --list-envs
python train.py --list-envs CartPole
```

**Debugger support**: When a debugger is attached, `train.py` automatically forces `n_envs=1` and `vectorization_mode='sync'` for reliable breakpoints, adjusting `batch_size` to remain compatible.

### Evaluation

```bash
# Play trained policy (auto-loads best/last checkpoint)
python run_play.py --run-id @last --episodes 5

# Launch Gradio inspector UI
python run_inspect.py --run-id @last --port 7860
```

### Publishing

Publish trained models to Hugging Face Hub:

```bash
# Publish latest run
python run_publish.py

# Publish specific run
python run_publish.py --run-id <ID>

# Publish to custom repo
python run_publish.py --repo user/repo --private
```

Requires `HF_TOKEN` environment variable or `huggingface-cli login`.

## Configuration

### YAML Configs

Configs live in `config/environments/*.yaml`. Each file defines base settings and per-algorithm variants:

```yaml
# Base settings with YAML anchors
_base: &base
  env_id: CartPole-v1
  eval_episodes: 10

ppo:
  <<: *base
  algo_id: ppo
  n_envs: 8
  max_timesteps: 1e5
  n_steps: 32
  batch_size: 256
  policy_lr: {start: 0.001, end: 0.0}   # Linear schedule from 0.001 to 0
  clip_range: {start: 0.2, end: 0.0}
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

**Config selection**:
- **CLI**: `python train.py <env>:<variant>` (e.g., `CartPole-v1:ppo`)
- **Python**: `load_config("CartPole-v1", "ppo")`

When `project_id` is omitted in YAML, it defaults to the filename stem.

### Key Configuration Fields

- **Environment**: `env_id`, `n_envs`, `vectorization_mode`, `env_wrappers`
- **Algorithm**: `algo_id` (ppo, reinforce), `policy` (mlp, cnn), `hidden_dims`
- **Training**: `max_timesteps`, `n_steps`, `batch_size`, `n_epochs`
- **Optimization**: `policy_lr`, `optimizer` (adam, adamw, sgd)
- **PPO-specific**: `clip_range`, `vf_coef`, `ent_coef`, `gae_lambda`
- **Atari**: `obs_type` (rgb, ram, objects)
- **Normalization**: `normalize_obs` (false, true/rolling, static), `normalize_reward`

### Schedules

Hyperparameters support dict-based linear schedules:

```yaml
policy_lr: {start: 0.001, end: 0.0}
clip_range: {start: 0.2, end: 0.05}
```

Control interpolation with `from`, `to`, and `schedule` keys. Values `<1` are fractions of `max_timesteps`, values `>1` are absolute steps.

### Fractional Batch Size

Batch size can be a fraction in (0, 1]:

```yaml
batch_size: 0.5  # 50% of rollout size (n_envs * n_steps)
```

Resolved as `floor(rollout_size * fraction)`, minimum 1, must evenly divide rollout size.

### Algorithm-Specific Options

**REINFORCE**:
- `policy_targets`: `returns` or `advantages`
- `returns_type`: `mc:rtg` (reward-to-go) or `mc:episode` (constant across episode)

**PPO + Replay (experimental)**:
- `replay_ratio`: off-policy minibatches per on-policy minibatch (0-8)
- `replay_buffer_size`: capacity in transitions
- `replay_is_clip`: importance sampling weight cap (default 10.0)

## Environment Wrappers

Register wrappers by name via `EnvWrapperRegistry` (see `gym_wrappers/__init__.py`):

- `PixelObservationWrapper`: Extract pixel observations
- `DiscreteEncoder`: Encode discrete observations (array, binary, onehot)
- `DiscreteActionSpaceRemapperWrapper`: Remap action space
- `PongV5_FeatureExtractor`, `PongV5_RewardShaper`: Pong-specific
- `MountainCarV0_RewardShaper`: Mountain Car reward shaping
- `CartPoleV1_RewardShaper`: CartPole reward shaping
- `VizDoom_RewardShaper`: VizDoom reward shaping

**Usage in YAML**:

```yaml
env_wrappers:
  - { id: PixelObservationWrapper, pixels_only: true }
  - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

## Runs and Outputs

### Directory Structure

Each training run creates `runs/<id>/` containing:

- `config.json`: Full configuration snapshot
- `checkpoints/*.ckpt`: Model checkpoints with embedded videos
- `best.ckpt`, `last.ckpt`: Symlinks to best/last checkpoints
- `metrics.csv`: Training metrics log
- `logs/`: Session logs
- `videos/`: Episode recordings
- `report.md`: End-of-training summary

The `runs/@last` symlink always points to the most recent run.

### Metrics and Logging

Metrics are printed to console and logged to W&B/CSV:

- **Training**: `train/roll/ep_rew/mean`, `train/roll/ep_len/mean`, `train/loss/*`
- **Evaluation**: `val/roll/ep_rew/mean`, `val/roll/ep_rew/best`
- **Optimization**: `train/opt/grads/norm/*`, `train/policy_lr`

The console logger displays:
- Precision/highlight rules from `config/metrics.yaml`
- Yellow highlighting for metrics outside configured bounds
- Inline ASCII sparklines showing recent trends (e.g., `█▇▇▆▅▄▃▂▁`)

### Video Capture

Videos are automatically recorded during evaluation epochs and uploaded to W&B. Episodes are stored in `runs/<id>/checkpoints/epoch=XX.mp4`.

## W&B Integration

### Automatic Dashboard Creation

Training automatically creates/updates a W&B workspace at the beginning of each run, selecting the current run across panels and printing the URL.

### Manual Dashboard Management

```bash
# Create/update dashboard (idempotent by default)
python scripts/setup_wandb_dashboard.py --entity <entity> --project <project>

# Force overwrite existing workspace
python scripts/setup_wandb_dashboard.py --overwrite

# Select latest run in dashboard panels
python scripts/setup_wandb_dashboard.py --select-latest
```

Requires W&B login (`wandb login`) or `WANDB_API_KEY` environment variable.

### W&B Sweeps

#### Local Sweeps

Run hyperparameter sweeps locally via W&B Agent:

```bash
# Create sweep
wandb sweep config/sweeps/cartpole_ppo_grid.yaml

# Run sweep agent locally
wandb agent <entity>/<project>/<sweep_id>
```

#### Distributed Sweeps (Modal AI)

Scale out sweeps across multiple cloud CPU instances using Modal AI:

```bash
# Install Modal dependencies (one-time)
pip install -e ".[modal]"
modal token new
modal secret create wandb-secret WANDB_API_KEY=<your-key>

# Create sweep and launch 10 Modal workers
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10

# Launch workers for existing sweep
python scripts/sweep_modal.py --sweep-id <sweep_id> --count 20

# Configure parallelism (50 runs, 5 per worker = 10 workers)
python scripts/sweep_modal.py --sweep-id <sweep_id> --count 50 --runs-per-worker 5
```

Each Modal worker runs on 2 CPUs with 4GB RAM and 1-hour timeout (configurable in `scripts/modal_sweep_runner.py`). See [scripts/README_MODAL.md](scripts/README_MODAL.md) for detailed documentation.

**Configuration**:
- Auto-detects sweep mode via `WANDB_SWEEP_ID` and merges `wandb.config` into main config
- Config fields map 1:1 to sweep parameters (e.g., `n_envs`, `policy_lr`, `clip_range`)
- Dict-based schedules like `{start: 0.001, end: 0.0}` are supported

**Example sweep specs**:
- Grid search: `config/sweeps/cartpole_ppo_grid.yaml`
- Bayesian optimization: `config/sweeps/cartpole_ppo_bayes.yaml`
- See [config/sweeps/README.md](config/sweeps/README.md) for sweep configuration guide

## Environment Variables

### Logging Control

```bash
# Suppress session log files and banner/config dumps
export VIBES_DISABLE_SESSION_LOGS=1

# Silence verbose checkpoint load prints
export VIBES_QUIET=1  # Alias: VIBES_DISABLE_CHECKPOINT_LOGS=1
```

## Testing

Run the test suite:

```bash
# All tests
pytest -q

# Specific test file
pytest tests/test_ppo.py -v

# Exclude slow tests
pytest -m "not slow" -q
```

## Development Tools

### Smoke Tests

Test all configurations briefly:

```bash
# Train all configs (default 100 steps each)
python scripts/smoke_all_configs.py

# Adjust timesteps
python scripts/smoke_all_configs.py --timesteps 50

# Filter environments
python scripts/smoke_all_configs.py --filter CartPole --limit 3
```

Prints PASS/FAIL with visual indicators and a summary. Uses small rollouts and disables video/W&B for speed.

### Benchmarking

```bash
# Benchmark vectorized environment FPS
python scripts/benchmark_vecenv_fps.py

# Benchmark rollout collectors
python scripts/benchmark_collectors.py

# Benchmark dataloaders
python scripts/benchmark_dataloaders.py
```

### Brax/JAX (Optional)

```bash
# Train Brax policy (requires jax[cuda12] or jax[cpu])
python scripts/brax_train_policy.py

# Evaluate Brax policy with Gymnasium wrapper
python scripts/brax_eval_policy.py
```

## Project Structure

```
agents/              # PPO, REINFORCE, base agent
loggers/             # Custom Lightning/CSV/console loggers
utils/               # Config, environment, models, rollouts, helpers
gym_wrappers/        # Wrapper registry + domain-specific wrappers
trainer_callbacks/   # Metrics, checkpointing, early stopping, videos
gym_envs/            # Custom environments (e.g., multi-armed bandits)
config/              # Environment YAML configs and metrics rules
runs/                # Training outputs (checkpoints, videos, logs)
scripts/             # Smoke tests, benchmarks, utilities
tests/               # Test suite
VIBES/               # Architecture guide, coding principles, task playbooks
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive guide for working with the codebase
- **[VIBES/ARCHITECTURE_GUIDE.md](VIBES/ARCHITECTURE_GUIDE.md)**: Detailed architecture documentation
- **[VIBES/CODING_PRINCIPLES.md](VIBES/CODING_PRINCIPLES.md)**: Coding style and principles

## License

MIT License - see [LICENSE](LICENSE) file for details.
