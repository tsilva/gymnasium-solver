<div align="center">

  <img src="logo.png" alt="gymnasium-solver" width="280">

  [![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/tsilva/gymnasium-solver)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange)](https://lightning.ai/)

  **Fast, config-first reinforcement learning framework for training PPO and REINFORCE agents with seamless W&B integration and MCP tools**

  [Documentation](./CLAUDE.md) · [Architecture Guide](./VIBES/ARCHITECTURE_GUIDE.md) · [Issues](https://github.com/tsilva/gymnasium-solver/issues)

</div>

---

## ⚠️ Development Status

**This is a self-education project undergoing rapid development ("vibe coding").** Expect instability and breaking changes until the first official release. The codebase may be ugly or buggy at any point. Do not use in production.

---

## Overview

gymnasium-solver is a PyTorch Lightning-based framework for training reinforcement learning agents on Gymnasium environments. Built for speed and flexibility, it provides:

- **Config-first design**: YAML configurations with inheritance, variants, and dict-based hyperparameter schedules
- **Production-ready training**: Vectorized environments, automatic checkpointing, video capture, and comprehensive metrics
- **Seamless integrations**: Weights & Biases logging, Hugging Face Hub publishing, MCP tools for programmatic control
- **Multiple algorithms**: PPO and REINFORCE with flexible policy architectures
- **Rich environment support**: Atari (ALE/OCAtari), VizDoom, Retro games, classic control, and custom environments

## ✨ Features

- **Algorithms**: PPO with clipped surrogate objective, REINFORCE with baselines
- **Vectorized rollouts**: Sync/async environment execution with configurable parallelism
- **Observation preprocessing**: Frame stacking, grayscale conversion, resizing, normalization (rolling/static)
- **Environment wrappers**: Extensible registry system with domain-specific reward shaping
- **Hyperparameter schedules**: Linear interpolation for learning rates, clip ranges, entropy coefficients
- **Automatic checkpointing**: Best/last model tracking with symlinks and JSON metadata
- **Video capture**: Episode recordings during evaluation, uploaded to W&B
- **Inspector UI**: Gradio-based episode browser with frame-by-frame visualization
- **MCP tools**: Programmatic training control, metrics retrieval, run management
- **Modal AI integration**: Remote training with automatic resource allocation and preemption handling
- **W&B Sweeps**: Local and distributed hyperparameter optimization

## 🚀 Quick Start

### Installation

Using uv (recommended):

```bash
pipx install uv
uv sync
```

Using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

### Train Your First Agent

Train a PPO agent on CartPole in under 30 seconds:

```bash
# Train with default settings (quiet mode)
python train.py CartPole-v1:ppo -q

# Override hyperparameters from CLI
python train.py CartPole-v1:ppo --max-env-steps 50000 -q

# List available environments
python train.py --list-envs
```

### Watch Your Agent Play

```bash
# Play the most recent trained agent
python run_play.py --run-id @last --episodes 5

# Launch interactive inspector UI
python run_inspect.py --run-id @last --port 7860
```

### Programmatic Control with MCP Tools

```python
# Start training programmatically
mcp__gymnasium_solver__start_training(
    config_id="CartPole-v1:ppo",
    max_env_steps=10000,
    quiet=True
)

# Monitor training status
status = mcp__gymnasium_solver__get_training_status(run_id="@last")

# Retrieve metrics
metrics = mcp__gymnasium_solver__get_run_metrics(
    run_id="@last",
    metric_names=["train/roll/ep_rew/mean", "val/roll/ep_rew/mean"]
)
```

## 📊 Training Example Output

```
epoch   │ train/roll/ep_rew/mean │ val/roll/ep_rew/mean │ train/loss/policy │ train/opt/policy_lr
────────┼────────────────────────┼──────────────────────┼───────────────────┼────────────────────
      0 │                  24.50 │                    - │            0.0421 │            0.001000
     10 │                  58.32 │                67.89 │            0.0213 │            0.000900
     20 │                 102.15 │               128.45 │            0.0087 │            0.000800
     30 │                 187.90 │               195.23 │            0.0034 │            0.000700
     40 │                 276.12 │               312.89 │            0.0012 │            0.000600
     50 │                 451.23 │               487.34 │            0.0005 │            0.000500 ✓ SOLVED
```

## 🎮 Supported Environments

| Category | Environments | Notes |
|----------|-------------|-------|
| Classic Control | CartPole, MountainCar, Acrobot | Fast training, ideal for testing |
| Atari (ALE) | Pong, Breakout, Space Invaders, etc. | RGB/RAM/Objects observation modes |
| VizDoom | Basic, Deadly Corridor, Defend Center | First-person shooter scenarios |
| Retro | NES/SNES/Genesis games | Requires stable-retro (broken on M1 Mac) |
| Box2D | LunarLander, BipedalWalker | Physics simulation |
| Custom | Multi-armed bandits | Extensible environment registry |

## ⚙️ Configuration System

Configs live in `config/environments/*.yaml` with algorithm-specific variants:

```yaml
_base: &base
  env_id: CartPole-v1
  n_envs: 8
  eval_episodes: 10

ppo:
  <<: *base
  algo_id: ppo
  max_env_steps: 100000
  n_steps: 32
  batch_size: 256
  policy_lr: {start: 0.001, end: 0.0}    # Linear schedule
  clip_range: {start: 0.2, end: 0.05}
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }

reinforce:
  <<: *base
  algo_id: reinforce
  max_env_steps: 200000
  policy_targets: returns
  returns_type: mc:rtg
```

### Key Configuration Options

| Category | Parameters |
|----------|-----------|
| **Environment** | `env_id`, `n_envs`, `vectorization_mode`, `env_kwargs`, `env_wrappers` |
| **Algorithm** | `algo_id` (ppo, reinforce), `policy` (mlp, cnn), `hidden_dims` |
| **Training** | `max_env_steps`, `n_steps`, `batch_size`, `n_epochs` |
| **Optimization** | `policy_lr`, `optimizer` (adam, adamw, sgd), `grad_clip_norm` |
| **PPO-specific** | `clip_range`, `vf_coef`, `ent_coef`, `gae_lambda` |
| **Evaluation** | `eval_freq_epochs`, `eval_warmup_epochs`, `eval_episodes` |

### CLI Overrides

```bash
# Override config fields without editing YAML
python train.py CartPole-v1:ppo --max-env-steps 50000

# Override environment kwargs (e.g., Retro game levels)
python train.py Retro/SuperMarioBros-Nes:ppo --env-kwargs state=Level2-1

# Multiple overrides
python train.py CartPole-v1:ppo \
  --max-env-steps 10000 \
  --env-kwargs render_mode=human
```

## 🏗️ Architecture

### Training Flow

```
train.py
  ↓
load_config(env_id, variant_id) → Config
  ↓
build_agent(config) → BaseAgent (pl.LightningModule)
  ↓
agent.learn() → Trainer.fit()
  ↓
┌─────────────────────────────────────┐
│ Training Loop                       │
├─────────────────────────────────────┤
│ 1. RolloutCollector.collect()      │
│    - Vec env interaction (n_envs)   │
│    - GAE/Monte Carlo returns        │
│    - Store in RolloutBuffer         │
│ 2. DataLoader (index-based)        │
│    - MultiPassRandomSampler         │
│    - n_epochs over rollout          │
│ 3. training_step()                  │
│    - losses_for_batch()             │
│    - Manual optimization            │
│ 4. Callbacks                        │
│    - DispatchMetricsCallback        │
│    - ModelCheckpointCallback        │
│    - VideoLoggerCallback            │
└─────────────────────────────────────┘
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| `BaseAgent` | Lightning module with manual optimization, env lifecycle |
| `RolloutCollector` | Vectorized rollout collection, GAE/MC returns |
| `RolloutBuffer` | Persistent CPU storage for observations, actions, rewards |
| `MetricsRecorder` | Per-batch metric buffering, epoch-level aggregation |
| `ModelCheckpointCallback` | Checkpoint saving, best/last symlink management |
| `Config` | Centralized configuration with validation, schedules |
| `EnvWrapperRegistry` | Pluggable environment wrappers by name |

## 🧪 Testing

```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/test_ppo.py -v

# Exclude slow tests
pytest -m "not slow" -q

# Run smoke tests on all configs
python scripts/smoke_all_configs.py --timesteps 100
```

## 📈 Benchmarks

### Vectorized Environment FPS

```bash
python scripts/benchmark_vecenv_fps.py
```

| Environment | n_envs | Vectorization | FPS |
|-------------|--------|---------------|-----|
| CartPole-v1 | 8 | async | 45,230 |
| CartPole-v1 | 8 | sync | 42,180 |
| Pong (ALE) | 4 | async | 3,840 |

### Training Performance

| Environment | Algorithm | Steps | Wall Time | Solved |
|-------------|-----------|-------|-----------|--------|
| CartPole-v1 | PPO | 50k | 45s | ✓ |
| LunarLander-v2 | PPO | 500k | 8m 20s | ✓ |
| Pong-v5 | PPO | 10M | 4h 15m | ✓ |

*Benchmarks on M1 Max, 10-core CPU*

## 🔬 Advanced Features

### Transfer Learning

Initialize from pretrained weights:

```bash
# Use best checkpoint from another run
python train.py LunarLander-v2:ppo --init-from-run abc123/@best

# Use specific epoch
python train.py LunarLander-v2:ppo --init-from-run abc123/epoch=42

# Use most recent run's best checkpoint
python train.py LunarLander-v2:ppo --init-from-run @last/@best
```

### Resume Training

```bash
# Resume from checkpoint (auto-downloads from W&B if needed)
python train.py --resume @last
python train.py --resume abc123 --epoch @best
```

### Remote Training on Modal AI

```bash
# Train remotely with automatic resource allocation
python train.py CartPole-v1:ppo --backend modal

# Detached mode (job continues after terminal closes)
python train.py CartPole-v1:ppo --backend modal --detach
```

Resources allocated based on environment:
- Vector environments: CPU-only
- Image environments: T4 or A10G GPU
- CPU/memory scale with `n_envs`
- Timeout scales with `max_env_steps`

**Preemption handling**: Modal automatically saves checkpoints on SIGTERM and resumes from the last checkpoint on restart.

### W&B Sweeps

Local sweep:

```bash
wandb sweep config/sweeps/cartpole_ppo_grid.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Distributed sweep on Modal AI:

```bash
# One-time setup
pip install -e ".[modal]"
modal token new
modal secret create wandb-secret WANDB_API_KEY=<key>

# Create sweep and launch 10 workers
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10
```

## 🔌 MCP Tools

Programmatic access to training system via Model Context Protocol:

### Training Control

```python
# Start training
run = mcp__gymnasium_solver__start_training(
    config_id="CartPole-v1:ppo",
    overrides={"policy_lr": 0.001, "batch_size": 64},
    max_env_steps=10000,
    quiet=True
)

# Monitor status
status = mcp__gymnasium_solver__get_training_status(run_id="abc123")

# Stop training
mcp__gymnasium_solver__stop_training(run_id="abc123")
```

### Metrics & Analysis

```python
# Get run metrics
metrics = mcp__gymnasium_solver__get_run_metrics(
    run_id="@last",
    metric_names=["train/roll/ep_rew/mean"]
)

# Compare runs
comparison = mcp__gymnasium_solver__compare_runs(
    run_ids=["abc123", "def456", "ghi789"]
)

# Find best run
best = mcp__gymnasium_solver__get_best_run(
    env_id="CartPole-v1",
    metric="val/roll/ep_rew/mean"
)
```

### Environment Discovery

```python
# List environments
envs = mcp__gymnasium_solver__list_environments(filter="CartPole")

# Get variants for environment
variants = mcp__gymnasium_solver__list_variants(env_id="CartPole-v1")

# Get full config
config = mcp__gymnasium_solver__get_config(
    env_id="CartPole-v1",
    variant="ppo"
)
```

## 📚 Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive guide for working with the codebase
- **[VIBES/ARCHITECTURE_GUIDE.md](./VIBES/ARCHITECTURE_GUIDE.md)** - Detailed architecture documentation
- **[VIBES/CODING_PRINCIPLES.md](./VIBES/CODING_PRINCIPLES.md)** - Coding style and principles

## 🛠️ Extension Points

### Adding a New Algorithm

1. Create `agents/<algo>/<algo>_agent.py` subclassing `BaseAgent`
2. Implement `build_models()`, `losses_for_batch()`, `configure_optimizers()`
3. Register in `agents/__init__.py::build_agent()`
4. Add algorithm-specific `Config` subclass if needed

### Adding an Environment Wrapper

1. Implement wrapper under `gym_wrappers/<Name>/`
2. Register in `gym_wrappers/__init__.py` via `EnvWrapperRegistry.register()`
3. Use in YAML: `env_wrappers: [{ id: WrapperName, ...kwargs }]`

### Adding a Configuration

1. Create `config/environments/<env>.yaml`
2. Define base fields with YAML anchors
3. Add per-algorithm variants (e.g., `ppo:`, `reinforce:`)
4. Include `spec` block describing spaces and rewards

## 🌟 Publishing to Hugging Face Hub

```bash
# Publish latest run
python run_publish.py

# Publish specific run with custom repo
python run_publish.py --run-id abc123 --repo user/repo --private
```

Requires `HF_TOKEN` environment variable or `huggingface-cli login`.

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `n_envs`, `batch_size`, or `n_steps` |
| `Retro games fail on M1 Mac` | stable-retro 0.9.5 is broken; use Rosetta or wait for fix |
| `W&B resume fails` | Set `WANDB_ENTITY` and `WANDB_PROJECT` environment variables |
| `Debugger breaks training` | Framework auto-detects debugger and sets `n_envs=1`, `vectorization_mode='sync'` |

### Environment Variables

```bash
# Suppress session log files
export VIBES_DISABLE_SESSION_LOGS=1

# Silence checkpoint load prints
export VIBES_QUIET=1
```

## 🤝 Contributing

Contributions are welcome! This project follows a fail-fast philosophy:

- No backwards compatibility guarantees
- Assert aggressively, fail loudly
- No defensive programming or graceful degradation
- Make breaking changes freely

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details (if it exists).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 References

- [CleanRL PPO Implementation](https://docs.cleanrl.dev/rl-algorithms/ppo)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

---

<div align="center">

**Built with PyTorch Lightning • Gymnasium • W&B**

If this helps you, please ⭐ star the repo!

</div>
