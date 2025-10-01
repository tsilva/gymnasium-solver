# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

gymnasium-solver is a fast, config-first reinforcement learning framework built on PyTorch Lightning, Gymnasium, and Stable-Baselines3. It trains PPO and REINFORCE agents with vectorized environments, video capture, and seamless W&B/Hugging Face Hub integration.

**Important**: This is a self-education project undergoing rapid development ("vibe coding"). Expect instability and breaking changes until the first official release.

## Common Commands

### Training
```bash
# Train with a specific config (env:variant format required)
python train.py CartPole-v1:ppo -q
python train.py CartPole-v1:reinforce -q

# Override max timesteps from CLI
python train.py CartPole-v1:ppo --max-timesteps 5000

# List available environments (optionally filtered)
python train.py --list-envs
python train.py --list-envs CartPole
```

### Testing
```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/test_ppo.py -v

# Run with markers
pytest -m "not slow" -q
```

### Evaluation & Inspection
```bash
# Play trained policy (auto-loads best/last checkpoint)
python run_play.py --run-id @last --episodes 5

# Launch Gradio inspector UI
python run_inspect.py --run-id @last --port 7860
```

### Publishing
```bash
# Publish to Hugging Face Hub (requires HF_TOKEN or huggingface-cli login)
python run_publish.py                    # latest run
python run_publish.py --run-id <ID>      # specific run
python run_publish.py --repo user/repo --private
```

### W&B Workspace Management
```bash
# Create/update W&B dashboard (idempotent by default)
python scripts/setup_wandb_dashboard.py --entity <entity> --project <project>

# Force overwrite existing workspace
python scripts/setup_wandb_dashboard.py --overwrite

# Select latest run in dashboard panels
python scripts/setup_wandb_dashboard.py --select-latest
```

### Smoke Tests
```bash
# Train all configs briefly (default 100 steps each)
python scripts/smoke_train_envs.py
python scripts/smoke_train_envs.py --timesteps 50 --filter CartPole

# Random-action environment smoke tests
python scripts/smoke_test_envs.py
python scripts/smoke_test_envs.py --filter Pong --limit 3
```

### Benchmarking
```bash
# Benchmark vectorized environment FPS
python scripts/benchmark_vecenv_fps.py

# Benchmark rollout collectors
python scripts/benchmark_collectors.py

# Benchmark dataloaders
python scripts/benchmark_dataloaders.py
```

### Brax/JAX Training (Optional Dependency)
```bash
# Train Brax policy (requires jax[cuda12] or jax[cpu])
python scripts/brax_train_policy.py

# Evaluate Brax policy with Gymnasium wrapper
python scripts/brax_eval_policy.py
```

## Architecture Overview

### Entry Point & Training Flow
- **`train.py`**: Accepts `<env>:<variant>` config specs (e.g., `CartPole-v1:ppo`), loads `Config` via `utils.config.load_config(env_id, variant_id)`, seeds via SB3, builds agent via `agents.build_agent()`, and calls `agent.learn()`.
- **Config override**: `--max-timesteps` injects into `config.max_timesteps` after sweep merges.
- **W&B Sweeps**: Auto-detected via `WANDB_SWEEP_ID` or `--wandb_sweep` flag. Merges `wandb.config` into main `Config` before training. Supports linear schedules like `lin_0.001`.
- **Debugger detection**: When a debugger is attached, `train.py` forces `n_envs=1`, `subproc=False`, and adjusts `batch_size` to remain compatible.

### Configuration System (`utils/config.py`)
- **Structure**: `Config` dataclass aggregates env, algo, rollout, model, optimization, eval, logging, and runtime settings.
- **Config files**: YAML files in `config/environments/*.yaml` with base fields at top (or under `_base` with YAML anchors) and per-algorithm variants nested below (e.g., `ppo:`, `reinforce:`).
- **Loading**: `load_config(env_id, variant_id)` requires both parameters. Callers must always provide a variant. The CLI enforces `env:variant` format.
- **Schedules**: Strings like `lin_0.001` are parsed into `*_schedule='linear'` plus numeric base. Control interpolation with `*_schedule_start_value`, `*_schedule_end_value`, `*_schedule_start`, and `*_schedule_end` (values `<1` are fractions of `max_timesteps`, values `>1` are absolute vec steps).
- **Fractional batch size**: When `batch_size` is in (0, 1], it's treated as a fraction of rollout size (`n_envs * n_steps`). Resolved to `floor(rollout_size * fraction)`, minimum 1, and must evenly divide rollout size.
- **Algo-specific subclasses**: `PPOConfig` enforces `clip_range > 0`; `REINFORCEConfig` validates `policy_targets` in {'returns', 'advantages'}.

### Environment Construction (`utils/environment.py`, `gym_wrappers/*`)
- **`build_env_from_config(config, **overrides)`**: Expands `Config` into kwargs for `build_env()` with seed/rendering overrides.
- **`build_env(...)`**: Builds vectorized environments with support for:
  - **ALE/Atari**: `obs_type` in {'rgb', 'ram', 'objects'} (objects uses OCAtari)
  - **VizDoom**: Matches `VizDoom-*` env_id
  - **Retro**: Matches `Retro/*` env_id (uses stable-retro)
  - **Multi-armed bandits**: Matches `Bandit-*` or `Bandit/<name>`
  - **Standard Gymnasium**: Everything else
- **Preprocessing**: Optional `GrayScaleObservation` (when `grayscale_obs` set), `ResizeObservation` (when `resize_obs` provided; `True` defaults to `(84, 84)`).
- **Normalization**:
  - **Observations**: `normalize_obs` accepts `false` (default), `true`/`'rolling'` (SB3 `VecNormalize`), or `'static'` (bounds-based).
  - **Rewards**: `normalize_reward: true` enables SB3 reward normalization (ignored when obs uses `'static'`).
- **Wrappers**: Registry system in `gym_wrappers/__init__.py`. YAML configs specify wrappers as `{ id: WrapperName, ...kwargs }`.
- **Wrapper classes**: Each base env wrapped with `EnvInfoWrapper`; vec env wrapped with `VecEnvInfoWrapper` exposing helpers like `.recorder()`, `get_return_threshold()`, `get_max_episode_steps()`, and `.is_rgb_env()`.

### Agent Architecture (`agents/base_agent.py`)
- **Base class**: `BaseAgent` is a `pl.LightningModule` with manual optimization (`automatic_optimization=False`).
- **Lifecycle**: Constructs train/val/test envs with offset seeds, builds `RolloutCollector`s for each stage, instantiates `MetricsRecorder`, `MetricsMonitor`, and `Run` object.
- **Training loop**: `train_dataloader()` collects rollout with `RolloutCollector`, builds index-collate `DataLoader` (`utils.dataloaders`), Lightning calls `training_step()` which delegates to `losses_for_batch()`.
- **Evaluation**: Validation/test hooks reuse stage-specific `RolloutCollector.evaluate_episodes(...)`, wrapping vec env with `env.recorder(...)` to optionally capture `epoch=XX.mp4` videos.
- **Schedules**: Auto-wires any `*_schedule` fields in `Config` to `HyperparameterSchedulerCallback`s. Special setters update both attribute and optimizer (e.g., `policy_lr`).
- **Checkpointing**: `ModelCheckpointCallback` writes `epoch=<idx>.ckpt` + JSON sidecar each eval epoch, maintains `best.*`/`last.*` symlinks.
- **Metrics**: Per-batch metrics buffered in recorder, flushed once per epoch by `DispatchMetricsCallback`. Tracks best episode rewards as `train/roll/ep_rew/best` and `val/roll/ep_rew/best`.

### Algorithms
- **PPO** (`agents/ppo/ppo_agent.py`):
  - Builds actor-critic via `utils.policy_factory.build_policy_from_env_and_config`.
  - Defaults to AdamW (configurable via `Config.optimizer`).
  - Logs gradient norms for `actor_head`, `critic_head`, and `trunk` before clipping.
  - Registers metric monitors for KL, clip fraction, explained variance with alert thresholds.
- **REINFORCE** (`agents/reinforce/reinforce_agent.py`):
  - Builds policy-only network via `utils.policy_factory.build_policy`.
  - Supports returns- or advantages-weighted policy targets (`config.policy_targets`).
  - Disables GAE in collector for classic REINFORCE behavior.
  - Monte Carlo return mode controlled by `Config.returns_type`: `mc:rtg` (reward-to-go, default) or `mc:episode` (constant across episode segment).

### Rollouts & Data Pipeline (`utils/rollouts.py`, `utils/dataloaders.py`)
- **`RolloutCollector`**: Collects `n_steps` across `n_envs` into persistent CPU `RolloutBuffer`. Supports GAE or Monte Carlo returns. Handles `TimeLimit.truncated` with bootstrapped values.
- **Shape preservation**: `flatten_slice_env_major` preserves image observations as CHW tensors, only flattens vectors/scalars. Returns:
  - Vectors/scalars: `(N*T, D)` or `(N*T, 1)`
  - Images (CHW): `(N*T, C, H, W)`
- **Evaluation**: `evaluate_episodes(n_episodes, deterministic, timeout_seconds)` gathers finished episodes only, balances per-env episode targets, updates running aggregates like `ep_rew_best`, `ep_len_mean`, `total_timesteps`, and `total_vec_steps`.
- **DataLoader**: `build_index_collate_loader_from_collector` creates `DataLoader` over indices with `MultiPassRandomSampler` to perform `n_epochs` passes without epoch resets.

### Models & Policies (`utils/models.py`, `utils/policy_factory.py`)
- **MLP**: `MLPActorCritic` (actor-critic) and `MLPPolicy` (policy-only) for discrete actions. Distributions are `Categorical` from logits.
- **CNN**: `CNNActorCritic` and `CNNPolicy` share `_CNNTrunk` (Conv2d stack + optional MLP). Expect CHW inputs; deterministic `_EnsureCHW` unflattens flat features.
- **Factory**: `policy_factory` routes based on `policy` string or custom module classes, forwards `policy_kwargs`.

### Trainer & Callbacks (`utils/trainer_factory.py`, `trainer_callbacks/*`)
- **Trainer**: `build_trainer` constructs PL `Trainer` with custom callbacks. Progress bar/checkpointing disabled.
- **Loggers**: `WandbLogger`, `CsvLightningLogger` (writes `metrics.csv`), `PrintMetricsLogger` (pretty console table).
- **Callbacks**:
  - **`DispatchMetricsCallback`**: Resets per-epoch buffers, aggregates recorder metrics, calls `log_dict`.
  - **`WarmupEvalCallback`**: Disables validation during warmup epochs, re-enables with configured cadence after.
  - **`ModelCheckpointCallback`**: Saves `epoch=XX.*` artifacts, refreshes `best.*`/`last.*` symlinks on fit end.
  - **`VideoLoggerCallback`**: Watches `runs/<id>/videos`, uploads new media to W&B on epoch end.
  - **`EarlyStoppingCallback`**: Supports stop conditions on total timesteps and train/eval reward thresholds.
  - **`EndOfTrainingReportCallback`**: Writes `report.md` summarizing training.

### Runs & Outputs (`utils/run.py`)
- **Structure**: `Run` creates `runs/<id>/`, ensures `checkpoints/`, manages `@last` symlink.
- **Artifacts**: Each run contains `config.json`, `checkpoints/*.ckpt`, `logs/`, `videos/`, `metrics.csv`, `run.log`.
- **Best/last symlinks**: `best.ckpt` and `last.ckpt` auto-updated by `ModelCheckpointCallback`.

### Metrics & Logging (`utils/metrics_*.py`, `loggers/*`, `config/metrics.yaml`)
- **Recorder**: `MetricsRecorder` buffers per-batch metrics, flushed once per epoch.
- **Monitor**: `MetricsMonitor` checks metric bounds/deltas against `config/metrics.yaml`, emits console warnings.
- **Console**: `PrintMetricsLogger` renders table with precision/highlight rules, inline ASCII sparklines per metric.
- **Namespaces**: Metric keys are `<train|val|test>/<metric>`. Use `utils.metrics_config` helpers for parsing.

## Key Conventions

### Config Selection
- **CLI**: Use `<env>:<variant>` format (e.g., `CartPole-v1:ppo`). Positional or via `--config_id`.
- **Python**: `load_config(env_id, variant_id)` requires both parameters. No fallback to first block.
- **Project ID**: When `project_id` is omitted in YAML, defaults to filename stem.

### Observation Shapes
- **Image observations**: Preserved as CHW tensors through rollout/DataLoader. CNN models consume CHW directly.
- **Vector/scalar observations**: Flattened to `(N*T, D)` or `(N*T, 1)`.
- **Q-Learning caveat**: Assumes discrete obs/action spaces. Do not auto-wrap discrete obs into vectors to avoid breaking tabular methods.

### Evaluation Cadence
- **Disabled**: When `eval_freq_epochs` is None or 0.
- **Warmup**: When `eval_warmup_epochs > 0`, all epochs up to and including warmup are skipped. Evaluation resumes on cadence grid (multiples of `eval_freq_epochs`) strictly after warmup.
- **Example**: `eval_warmup_epochs=50`, `eval_freq_epochs=15` → first eval at epoch 60 (epoch_idx=59).

### Environment-Specific Behavior
- **Retro**: Evaluation/test envs created lazily with `n_envs=1` to avoid multi-emulator errors. `evaluate_episodes` stops at requested episode count.
- **VizDoom**: Auto-discovers `.cfg`/`.wad` from installed package or `VIZDOOM_SCENARIOS_DIR` or `env_kwargs.config_path`.

### Environment Variables
- **`VIBES_DISABLE_SESSION_LOGS=1`**: Suppress session log files and banner/config dumps to `logs/`.
- **`VIBES_QUIET=1`**: Silence verbose checkpoint load prints (alias: `VIBES_DISABLE_CHECKPOINT_LOGS=1`).

## Extension Points

### Adding a New Algorithm
1. Create `agents/<algo>/<algo>_agent.py` subclassing `BaseAgent`.
2. Implement `build_models`, `losses_for_batch`, `configure_optimizers`.
3. Wire dispatch in `agents/__init__.py::build_agent`.
4. Add algo-specific `Config` subclass if needed (e.g., `PPOConfig`, `REINFORCEConfig`).

### Adding an Environment Wrapper
1. Implement under `gym_wrappers/<Name>/...`.
2. Register in `gym_wrappers/__init__.py` via `EnvWrapperRegistry.register`.
3. Use in YAML: `env_wrappers: [{ id: WrapperName, ...kwargs }]`.

### Adding a Config
1. Create or update `config/environments/<env>.yaml`.
2. Define base fields at top or under `_base` with YAML anchors.
3. Add per-algorithm variants as nested keys (e.g., `ppo:`, `reinforce:`).
4. Embed `spec` block describing action/observation spaces, rewards, metadata.
5. Use linear schedules like `lin_0.001` for hyperparameters.

### Metrics & Logging
1. Adjust `config/metrics.yaml` for precision/highlight rules.
2. Register additional `MetricsMonitor` callbacks in agents when alerts are needed.
3. Continue logging through `MetricsRecorder` buffers to keep Lightning stream consistent.

## Important Files & Directories

- **`train.py`**: Main training entry point.
- **`run_play.py`**: Play trained policy with rendering.
- **`run_inspect.py`**: Gradio episode browser UI.
- **`run_publish.py`**: Publish to Hugging Face Hub.
- **`agents/`**: PPO, REINFORCE, base agent.
- **`utils/`**: Config, env, rollout, dataloaders, models, metrics, helpers.
- **`gym_wrappers/`**: Wrapper registry + domain-specific wrappers.
- **`trainer_callbacks/`**: Metrics dispatch, warmup, checkpointing, early stopping, videos.
- **`gym_envs/`**: Custom environment implementations (e.g., MultiArmedBandit).
- **`config/environments/`**: YAML configs per environment.
- **`config/metrics.yaml`**: Metric precision/highlight rules.
- **`config/sweeps/`**: W&B sweep specs (grid, Bayesian).
- **`runs/`**: Training outputs (config, checkpoints, videos, logs, metrics).
- **`VIBES/`**: Agent helper docs (Architecture Guide, Coding Principles, task playbooks).
- **`VIBES/ARCHITECTURE_GUIDE.md`**: Detailed architecture documentation.
- **`VIBES/CODING_PRINCIPLES.md`**: Coding style and principles (fail-fast, no defensive programming, assert aggressively).
- **`AGENTS.md`**: Workspace rules for autonomous agents.

## Agent Task System

The codebase includes task playbooks under `VIBES/tasks/` for common maintenance operations:
- `!TASK: audit_bugs` - Find and document functional defects
- `!TASK: audit_static_analysis` - Identify latent defects
- `!TASK: cleanup_dead_code` - Remove unused code
- `!TASK: cleanup_dry_file` - Reduce duplication within modules
- `!TASK: audit_dryness` - Identify copy-paste logic
- `!TASK: audit_encapsulation` - Find shared behavior that can be encapsulated
- `!TASK: audit_concerns` - Find separation of concerns issues
- `!TASK: audit_config_consistency` - Validate config inconsistencies
- `!TASK: audit_dependency_health` - Assess dependency state
- `!TASK: tune_hyperparams` - Iteratively adjust hyperparameters
- `!TASK: update_docs_architecture` - Keep Architecture Guide accurate
- `!TASK: update_docs_coding_from_diff` - Infer coding preferences from changes
- `!TASK: update_docs_readme` - Keep README concise and accurate
- `!TASK: audit_tests` - Exercise test suites and document failures
- `!TASK: test_upgrade` - Strengthen test suite for high-risk modules
- `!TASK: cleanup_imports` - Streamline imports

**Usage**: `run task: <name>` maps to `VIBES/tasks/<name>.md`. Follow instructions in AGENTS.md when executing tasks.

## Important Principles

### Fail-Fast Philosophy (from CODING_PRINCIPLES.md)
- **Never preserve backwards compatibility**: Make breaking changes and force users to adapt.
- **No defensive programming**: Don't add safety checks, validation layers, or compatibility layers. Let the system fail if used incorrectly.
- **Assert aggressively**: Use assertions for all assumptions, preconditions, and invariants. Let the program crash if anything is unexpected.
- **No exception handling**: Never catch exceptions unless absolutely required by the API. Let errors propagate up and crash the program.
- **Explicit error propagation**: Never use broad exception handlers (`except Exception:`) or bare `except:`. Catch only specific, expected exceptions and re-raise with context.
- **No graceful degradation**: If something can't work correctly, fail immediately rather than falling back to partial functionality.

### Agent Working Guidelines (from AGENTS.md)
- **Read first**: Read `VIBES/ARCHITECTURE_GUIDE.md`, `VIBES/CODING_PRINCIPLES.md`, and `README.md` before making changes.
- **Minimal diffs**: Change only what's necessary; avoid drive-by refactors.
- **Root-cause-first**: Trace failures to true cause, plan minimal fix, avoid symptom patches.
- **Preserve formatting**: Keep indentation style and width; don't reflow unrelated lines.
- **Tests**: Add or update tests when behavior changes or is newly added.
- **After editing**: Validate imports/types, run tests/build if feasible, summarize changes concisely.