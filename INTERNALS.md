## Internals

High-signal reference for maintainers and agents. Read this before making changes.

### Top-level flow
- **Entry point**: `train.py` expects `--config_id "<env>:<variant>"` (e.g., `CartPole-v1:ppo`) and optional `-q/--quiet`; it splits env/variant, loads `Config` via `utils.config.load_config(config_id, variant_id)`, sets the global seed via `stable_baselines3.common.utils.set_random_seed`, creates the agent (`agents.create_agent`), and calls `agent.learn()`.
- **Agent lifecycle**: `agents/base_agent.BaseAgent` constructs envs, models, collectors, and the PyTorch Lightning Trainer; subclasses implement `create_models`, `losses_for_batch`, and optimizer.
- **Training loop**: `BaseAgent.train_dataloader()` collects a rollout with `utils.rollouts.RolloutCollector`, builds an index-collate `DataLoader` (`utils.dataloaders`), and Lightning calls `training_step()` which delegates to `losses_for_batch()`; optim is manual (`automatic_optimization=False`).
- **Eval**: Validation loop is procedural in hooks; `utils.evaluation.evaluate_policy` runs N episodes on a vectorized env; videos recorded via `VecVideoRecorder` wrapper and `trainer_callbacks.VideoLoggerCallback` watches media dir.
  - For Retro environments (stable-retro), evaluation/test envs are created lazily with `n_envs=1` to avoid multi-emulator-per-process errors, and `evaluate_policy` uses a hard cap on per-episode steps (currently 1,000 for validation; 2,000 for final video) to prevent very long episodes.
- **Runs**: `utils.run_manager.RunManager` creates `runs/<id>/`, symlinks `runs/@latest-run`, and manages paths; `BaseAgent` dumps `config.json` and writes `run.log` via `utils.logging.stream_output_to_log`.
- **Checkpoints**: `trainer_callbacks.ModelCheckpointCallback` writes `best.ckpt`/`last.ckpt`. Resume is enabled by passing `resume=True` to the callback and is gated by an optional `config.resume` flag when present.

### Configuration model (`utils/config.py`)
- `Config` dataclass aggregates env, algo, rollout, model, optimization, eval, logging, and runtime settings. The loader instantiates an algo-specific subclass based on `algo_id`.
- `load_from_yaml(config_id, variant_id)`: loads from `config/environments/*.yaml` using the new per-file format (base fields at the root + per-variant sections like `ppo:`). Schedules like `lin_0.001` are parsed into `*_schedule='linear'` and the numeric base. For new-style environment files, when `project_id` is not specified, it defaults to the YAML filename (stem).
- Variant selection: when `variant_id` is omitted, the loader auto-selects a default variant for the given project. Preference order is `ppo`, then `reinforce`, then `qlearning` when available; otherwise the first variant in lexical order is chosen. Note: the current CLI (`train.py`) still requires `env:variant` and does not exercise this default.
- Fractional batch size: when `batch_size` is a float in (0, 1], it is interpreted as a fraction of the rollout size (`n_envs * n_steps`). The loader computes `floor(rollout_size * fraction)` with a minimum of 1; no divisibility assertion is enforced.

Algo-specific config subclasses:
- `PPOConfig`: enforces `clip_range > 0`.
- `REINFORCEConfig`: validates `policy_targets` in {'returns','advantages'}.
- `QLearningConfig`: retains base validations; ensures `n_envs >= 1`.

### Environments (`utils/environment.py`, `gym_wrappers/*`)
- `build_env(env_id, n_envs, seed, subproc, obs_type, frame_stack, norm_obs, env_wrappers, env_kwargs, render_mode, record_video, record_video_kwargs)` builds a vectorized env:
  - ALE via `ale_py` with optional `obs_type in {'rgb','ram','objects'}` (objects uses OCAtari).
  - Optional base-env preprocessing (Gymnasium wrappers): `GrayScaleObservation` when `grayscale_obs` is set, and `ResizeObservation` when `resize_obs` is provided. `resize_obs` accepts `(width, height)`; `True` is treated as `(84, 84)` for convenience.
  - VizDoom via `gym_wrappers.vizdoom.VizDoomEnv` when `env_id` matches `VizDoom-*`.
  - Retro via `stable-retro` when `env_id` matches `Retro/*`.
  - Standard Gymnasium otherwise.
  - Applies registry wrappers from YAML: `EnvWrapperRegistry.apply` with `{ id: WrapperName, ... }` specs.
- Observation normalization: `norm_obs == 'static'` uses `VecNormalizeStatic`; `norm_obs == 'rolling'` uses `VecNormalize`. Note: the config field is `normalize_obs` (bool) but the builder currently expects these string values; a boolean `True/False` will not enable normalization.
  - Optional `VecFrameStack` and `VecVideoRecorder`.
  - Always wraps with `VecInfoWrapper` for metadata and shape helpers.
- Wrapper registry: `gym_wrappers.__init__` registers `PixelObservationWrapper` and domain wrappers like `PongV5_FeatureExtractor`, reward shapers, etc.

### Agents
- `BaseAgent` (LightningModule):
  - Creates `train_env`, `validation_env`, `test_env` with different seeds and video settings; builds `RolloutCollector`s for each.
  - Manual optimization in `training_step`; metrics buffered via `utils.metrics_buffer.MetricsBuffer` and printed with `trainer_callbacks.PrintMetricsCallback` using `utils.metrics` rules.
  - Tracks best episode rewards: exposes `train/ep_rew_best` and `eval/ep_rew_best` computed from the running best of `*/ep_rew_mean` across epochs. These appear in `metrics.csv` and the console table, highlighted in blue (highlight rules configurable via `config/metrics.yaml` under `_global.highlight`).
  - Evaluation cadence controlled by `Config.eval_freq_epochs` and `eval_warmup_epochs`; `on_validation_epoch_start/validation_step` drive `evaluate_policy` and video recording.
  - Schedules: learning rate and PPO clip range with linear decay based on progress `total_steps / max_timesteps`.
  - Logging: W&B via `WandbLogger` with project derived from `env_id`; defines step metric `train/total_timesteps`.
- `agents/ppo.PPO`:
  - Policy via `utils.policy_factory.create_actor_critic_policy` using MLP or CNN heads; computes PPO losses, metrics, and uses Adam(eps=1e-5).
  - Linear schedule for `clip_range` if `clip_range_schedule == 'linear'`.
  - Logs gradient norms per component under `train/grad_norm/*` after backward and before clipping/step: `actor_head`, `critic_head`, and `trunk`.
- `agents/reinforce.REINFORCE`:
  - Policy-only via `create_policy`; can use returns or advantages as policy targets (`config.policy_targets`), with GAE disabled in the collector for classic REINFORCE behavior.
  - Logs policy diagnostics including `entropy`, and PPO-style KL indicators `kl_div` and `approx_kl` computed from rollout (old) vs current log-probs for the taken actions.
  - Monte Carlo return mode is controlled by the loader/collector via `returns_type` (e.g., `mc:rtg` as default or `mc:episode` to scale all timesteps by the episode return).
- `agents/qlearning.QLearning`:
  - Tabular Q-table on discrete obs/action spaces; custom `QLearningPolicyModel` with epsilon decay; no optimizer.

### Rollouts and data pipeline (`utils/rollouts.py`, `utils/dataloaders.py`)
- `RolloutCollector` collects `n_steps` across `n_envs` and stores into a persistent CPU `RolloutBuffer` to minimize allocs.
- Supports GAE or Monte Carlo returns; handles `TimeLimit.truncated` with bootstrapped values; maintains rolling windows for ep stats and FPS.
- `flatten_slice_env_major` preserves image observations in CHW shape and only flattens vectors/scalars. Shapes returned:
  - Vectors/scalars: `(N*T, D)` or `(N*T, 1)`; Images (CHW): `(N*T, C, H, W)`.
- Terminal observations for time-limit truncations are coerced to the buffer's CHW shape before value bootstrapping.
- `build_index_collate_loader_from_collector` creates a `DataLoader` over indices with `MultiPassRandomSampler` to perform `n_epochs` passes over the same rollout without epoch resets.

### Models and policies (`utils/models.py`, `utils/policy_factory.py`)
- MLP actor-critic (`MLPActorCritic`) and policy-only (`MLPPolicy`) for discrete actions; distributions are `Categorical` from logits. `MLPPolicy` accepts either `input_dim`/`output_dim` or `input_shape`/`output_shape` for factory compatibility.
- CNN variants (`CNNActorCritic`, `CNNPolicy`) share a reusable `_CNNTrunk` (Conv2d stack + optional MLP). Models expect CHW inputs; a deterministic `_EnsureCHW` unflattens when given flat features (no HWC/CHW heuristics).
- `policy_factory` routes based on `policy` string or custom module classes and forwards `policy_kwargs`.

### Trainer and callbacks (`utils/trainer_factory.py`, `trainer_callbacks/*`)
- `build_trainer` constructs a PL `Trainer` with validation cadence controlled externally; progress bar/checkpointing disabled in favor of custom callbacks.
- Callbacks include:
  - `CSVMetricsLoggerCallback`: writes `metrics.csv` under run dir.
  - `PrintMetricsCallback`: pretty prints metrics with precision/delta rules from `utils.metrics`.
  - `HyperparamSyncCallback`: hot-reloads config changes from file; can enable manual control.
  - `ModelCheckpointCallback`: saves best/last; supports resume.
  - `VideoLoggerCallback`: uploads videos from run media dir.
  - `EarlyStoppingCallback`: supports stop on timesteps and reward thresholds (train/eval).
  - `EndOfTrainingReportCallback`: writes `report.md`.

### Scripts and tools
- `play.py`: loads best/last checkpoint from `runs/<id>/checkpoints` and steps env for rendering.
- `inspector.py`: Gradio UI to browse episodes for a run.
  - Visualizes raw rendered frames for all envs.
  - Shows processed observations and frame stacks when available:
    - Image observations: processed single frame plus tiled frame-stack grid.
    - Vector observations with `frame_stack > 1` (e.g., CartPole-v1 MLP): visualizes each stacked frame as a grayscale bar and tiles them into a grid.
- `publish.py`: push artifacts/videos to Hugging Face Hub.
- `scripts/*`: smoke tests, benchmarks, helpers (rendering, dataset checks, etc.).

### Testing
- `tests/` cover models, collectors, config loading, run manager symlink, callbacks, PPO behavior/integration, and wrappers.

### Extension points
- **Add an algorithm**: create `agents/<algo>.py` subclassing `BaseAgent`; implement `create_models`, `losses_for_batch`, `configure_optimizers`; wire `create_agent` in `agents/__init__.py`.
- **Add an env wrapper**: implement under `gym_wrappers/<Name>/...` and register in `gym_wrappers/__init__.py` or via `EnvWrapperRegistry.register`.
- **Add config**: create a new block in `config/environments/*.yaml` with `inherits` and fields; prefer schedules like `lin_0.001`.
- **Metrics/logging**: extend `utils.metrics` rules or callbacks; prefer buffering via `MetricsBuffer` to avoid Lightning overhead.

### Conventions and gotchas
- Observations are flattened only for vectors/scalars; image observations are preserved as CHW tensors through the rollout/DataLoader path. CNN models consume CHW directly (a small trunk handles shape consistency).
- Q-Learning assumes discrete obs/action spaces; do not auto-wrap discrete obs into vectors in `build_env` to avoid breaking tabular methods.
- When `eval_freq_epochs` is None or 0, validation is disabled; otherwise cadence is enforced via PL controls and warmup gates.
- Use `config.max_timesteps` for progress-based schedules; ensure it’s set for linear decays to have effect.
- Training CLI uses `--config_id "<env>:<variant>"`; underscore-only IDs like `Env_Variant` are not parsed by `train.py`. The config loader itself supports fully qualified IDs (e.g., `CartPole-v1_ppo`) and default-variant selection, but the current CLI does not.
- Env normalization: pass `normalize_obs: 'static'|'rolling'` for effect; boolean `True/False` does not enable normalization in `build_env`.

### Directory layout
```
agents/           # PPO, REINFORCE, QLearning, base
utils/            # config, env, rollout, dataloaders, models, logging, metrics, etc.
gym_wrappers/     # registry + wrappers (feature extractors, reward shaping, pixels)
trainer_callbacks/# logging, early stopping, checkpointing, hyperparam sync, videos
config/           # environment YAML configs
runs/             # run outputs: config.json, checkpoints/, videos/, metrics.csv, run.log
```
