## Internals

High-signal reference for maintainers and agents. Read this before making changes.

### Top-level flow
- **Entry point**: `train.py` parses CLI, loads `Config` via `utils.config.load_config`, sets seeds, creates agent (`agents.create_agent`), calls `agent.learn()`.
- **Agent lifecycle**: `agents/base_agent.BaseAgent` constructs envs, models, collectors, and the PyTorch Lightning Trainer; subclasses implement `create_models`, `losses_for_batch`, and optimizer.
- **Training loop**: `BaseAgent.train_dataloader()` collects a rollout with `utils.rollouts.RolloutCollector`, builds an index-collate `DataLoader` (`utils.dataloaders`), and Lightning calls `training_step()` which delegates to `losses_for_batch()`; optim is manual (`automatic_optimization=False`).
- **Eval**: Validation loop is procedural in hooks; `utils.evaluation.evaluate_policy` runs N episodes on a vectorized env; videos recorded via `VecVideoRecorder` wrapper and `trainer_callbacks.VideoLoggerCallback` watches media dir.
  - For Retro environments (stable-retro), evaluation/test envs are created lazily with `n_envs=1` to avoid multi-emulator-per-process errors, and `evaluate_policy` uses a hard cap on per-episode steps (currently 1,000 for validation; 2,000 for final video) to prevent very long episodes.
- **Runs**: `utils.run_manager.RunManager` creates `runs/<id>/`, symlinks `runs/latest-run`, and manages paths; `BaseAgent` dumps `config.json` and writes `run.log` via `utils.logging.stream_output_to_log`.
- **Checkpoints**: `trainer_callbacks.ModelCheckpointCallback` writes `best.ckpt`/`last.ckpt`; resume controlled by `Config.resume`.

### Configuration model (`utils/config.py`)
- `Config` dataclass aggregates env, algo, rollout, model, optimization, eval, logging, and runtime settings.
- `load_from_yaml(config_id, algo_id=None)`: loads from `config/environments/*.yaml`, supporting both the legacy multi-block format with `inherits` and the new per-file format (base at root + per-variant sections like `ppo:`). Parses schedules like `lin_0.001` into `*_schedule = 'linear'` and numeric base. For new-style environment files, when `project_id` is not specified, it defaults to the YAML filename (stem).
- Legacy loader `_load_from_legacy_config` remains for `config/hyperparams/<algo>.yaml`.
- Key derived behaviors: evaluation defaults when `eval_freq_epochs` set; RLZoo-style `normalize` mapped to `normalize_obs/reward`; `policy` in {'mlp','cnn'}; validation via `_compute_validation_controls` helpers.

### Environments (`utils/environment.py`, `gym_wrappers/*`)
- `build_env(env_id, n_envs, seed, subproc, obs_type, frame_stack, norm_obs, env_wrappers, env_kwargs, render_mode, record_video, record_video_kwargs)` builds a vectorized env:
  - ALE via `ale_py` with optional `obs_type in {'rgb','ram','objects'}` (objects uses OCAtari).
  - For ALE RGB, applies Gymnasium's `AtariPreprocessing` (grayscale+resize+frameskip). When this is active, manual `resize_obs`/`grayscale_obs` flags are ignored to avoid conflicts.
  - VizDoom Deadly Corridor via `gym_wrappers.vizdoom_deadly_corridor.VizDoomDeadlyCorridorEnv` when `env_id` matches.
  - Standard Gymnasium otherwise.
  - Applies registry wrappers from YAML: `EnvWrapperRegistry.apply` with `{ id: WrapperName, ... }` specs.
  - Optional `VecNormalize`, `VecFrameStack`, and `VecVideoRecorder`.
  - Always wraps with `VecInfoWrapper` for metadata and shape helpers.
- Wrapper registry: `gym_wrappers.__init__` registers `PixelObservationWrapper`, grayscale/resize aliases, plus domain wrappers like `PongV5_FeatureExtractor`, reward shapers, etc.

### Agents
- `BaseAgent` (LightningModule):
  - Creates `train_env`, `validation_env`, `test_env` with different seeds and video settings; builds `RolloutCollector`s for each.
  - Manual optimization in `training_step`; metrics buffered via `utils.metrics_buffer.MetricsBuffer` and printed with `trainer_callbacks.PrintMetricsCallback` using `utils.metrics` rules.
  - Evaluation cadence controlled by `Config.eval_freq_epochs` and `eval_warmup_epochs`; `on_validation_epoch_start/validation_step` drive `evaluate_policy` and video recording.
  - Schedules: learning rate and PPO clip range with linear decay based on progress `total_steps / n_timesteps`.
  - Logging: W&B via `WandbLogger` with project derived from `env_id`; defines step metric `train/total_timesteps`.
- `agents/ppo.PPO`:
  - Policy via `utils.policy_factory.create_actor_critic_policy` using MLP or CNN heads; computes PPO losses, metrics, and uses Adam(eps=1e-5).
  - Linear schedule for `clip_range` if `clip_range_schedule == 'linear'`.
- `agents/reinforce.REINFORCE`:
  - Policy-only via `create_policy`; can use returns or advantages as baseline; disables GAE in collector.
- `agents/qlearning.QLearning`:
  - Tabular Q-table on discrete obs/action spaces; custom `QLearningPolicyModel` with epsilon decay; no optimizer.

### Rollouts and data pipeline (`utils/rollouts.py`, `utils/dataloaders.py`)
- `RolloutCollector` collects `n_steps` across `n_envs` and stores into a persistent CPU `RolloutBuffer` to minimize allocs.
- Supports GAE or Monte Carlo returns; handles `TimeLimit.truncated` with bootstrapped values; maintains rolling windows for ep stats and FPS.
- Returns flattened env-major tensors via `flatten_slice_env_major` to feed training.
- `build_index_collate_loader_from_collector` creates a `DataLoader` over indices with `MultiPassRandomSampler` to perform `n_epochs` passes over the same rollout without epoch resets.

### Models and policies (`utils/models.py`, `utils/policy_factory.py`)
- MLP actor-critic (`ActorCritic`) and policy-only (`MLPPolicy`) for discrete actions; distributions are `Categorical` from logits.
- CNN variants (`CNNActorCritic`, `CNNPolicyOnly`) reshape flat obs back to image tensors based on inferred HWC; `NatureCNN` feature extractor; configurable via `policy_kwargs`.
- `policy_factory` routes based on `policy` string or custom module classes; infers image shapes from `observation_space`.

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
- `inspect.py` / `inspector_app.py`: Gradio UI to browse episodes for a run.
  - Visualizes raw rendered frames for all envs.
  - When observations are image-like, also shows processed observations and frame stacks (if configured).
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
- Observations are flattened before reaching models; CNNs must reshape using inferred HWC.
- Q-Learning assumes discrete obs/action spaces; do not auto-wrap discrete obs into vectors in `build_env` to avoid breaking tabular methods.
- When `eval_freq_epochs` is None, validation is disabled; otherwise cadence is enforced via PL controls and warmup gates.
- Use `config.n_timesteps` for progress-based schedules; ensure it’s set for linear decays to have effect.

### Directory layout
```
agents/           # PPO, REINFORCE, QLearning, base
utils/            # config, env, rollout, dataloaders, models, logging, metrics, etc.
gym_wrappers/     # registry + wrappers (feature extractors, reward shaping, pixels)
trainer_callbacks/# logging, early stopping, checkpointing, hyperparam sync, videos
config/           # environment YAML configs
runs/             # run outputs: config.json, checkpoints/, videos/, metrics.csv, run.log
```
