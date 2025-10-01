## Architecture Guide

High-signal reference for maintainers and agents. Read this alongside `VIBES/CODING_PRINCIPLES.md` before making changes.

### IO Helpers
- JSON and YAML file IO is centralized in `utils/io.py`.
  - Use `read_json(path)` / `write_json(path, data, indent=2, ensure_ascii=False, ...)`.
  - Use `read_yaml(path)` for YAML reads (write helpers intentionally omitted).
  - All functions use UTF-8 encoding by default for both reads and writes.

### Top-level flow
- **Entry point**: `train.py` accepts a positional config spec `"<env>:<variant>"` (e.g., `CartPole-v1:ppo`) or `--config_id` plus optional `-q/--quiet`; it splits env/variant, loads a `Config` via `utils.config.load_config(env_id, variant_id)`, seeds via `stable_baselines3.common.utils.set_random_seed`, builds the agent (`agents.build_agent`), and runs `agent.learn()`. When neither positional nor flag input is provided it defaults to `Bandit-v0:ppo`. Immediately after initializing the W&B run, it calls `utils.wandb_workspace.create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)` so the dashboard is available during training and prints the URL when available.
- **CLI overrides**: Passing `--max-steps` (aliases: `--max-timesteps` / `--max_timesteps`) from the CLI injects the requested integer into `config.max_timesteps` after sweep merges, letting you clamp training length without editing YAML.
- **W&B sweeps**: When launched by a W&B Agent (auto-detected via `WANDB_SWEEP_ID`) or with `--wandb_sweep`, `train.py` calls `wandb.init(config=asdict(config))` early and merges `wandb.config` into the main `Config` before creating the agent. Schedule strings like `lin_0.001` are parsed, and fractional `batch_size` in (0, 1] is resolved against `n_envs * n_steps` post-override.
- **Agent lifecycle**: `agents/base_agent.BaseAgent` constructs envs, models, collectors, and the PyTorch Lightning Trainer; subclasses implement `build_models`, `losses_for_batch`, and optimizer.
- **Training loop**: `BaseAgent.train_dataloader()` collects a rollout with `utils.rollouts.RolloutCollector`, builds an index-collate `DataLoader` (`utils.dataloaders`), and Lightning calls `training_step()` which delegates to `losses_for_batch()`; optim is manual (`automatic_optimization=False`).
- **Eval**: Validation/test hooks reuse each stage's `RolloutCollector.evaluate_episodes(...)`, wrapping the vec env with `env.recorder(...)` (from `VecEnvInfoWrapper`) to optionally capture `epoch=XX.mp4` videos in `runs/<id>/checkpoints/`. Per-epoch metrics flow through the same recorder/logging path as training, and `trainer_callbacks.VideoLoggerCallback` later uploads media from `runs/<id>/videos` when present.
  - For Retro environments (stable-retro), evaluation/test envs are created lazily with `n_envs=1` to avoid multi-emulator-per-process errors, and `evaluate_episodes` stops at the requested episode count (the collector balances episode targets across vec ranks) to prevent overly long runs.
- **Runs**: `utils.run.Run` creates `runs/<id>/`, ensures `checkpoints/`, manages the `@last` symlink, and exposes helpers to create paths; `BaseAgent` dumps `config.json` and writes `run.log` via `utils.logging.stream_output_to_log`.
- **Checkpoints**: `trainer_callbacks.ModelCheckpointCallback` writes `epoch=<idx>.ckpt` plus a JSON metrics sidecar each eval epoch, tracks the best metric value, and refreshes `best.*` / `last.*` symlinks at the end of training so helpers (e.g., `play.py`) always see the latest artifacts.

### Configuration model (`utils/config.py`)
- `Config` dataclass aggregates env, algo, rollout, model, optimization, eval, logging, and runtime settings. The loader instantiates an algo-specific subclass based on `algo_id`.
- Environment YAMLs embed a `spec` mapping (action/observation metadata, rewards, versions); the loader attaches it to `Config.spec` so `EnvInfoWrapper` can expose accurate ranges even when variants override the spec (e.g., Pong objects remaps actions).
- `load_config(env_id, variant_id)` delegates to `Config.build_from_yaml(...)`, loading from `config/environments/*.yaml` (base fields at the top or under `_base` with YAML anchors; per-variant blocks like `ppo:` nested below and may `<<: *base`). Schedule strings like `lin_0.001` are parsed into `*_schedule='linear'` plus the numeric base. When `project_id` is omitted in YAML it defaults to the filename stem.
- Variant selection: callers must provide a variant id; the loader builds `<project>_<variant>` keys and raises if the requested pair is missing. The CLI enforces the `env:variant` form and falls back to `Bandit-v0:ppo` only when no config is supplied on the command line.
- Fractional batch size: when `batch_size` is a float in (0, 1], it is interpreted as a fraction of the rollout size (`n_envs * n_steps`). The loader computes `floor(rollout_size * fraction)` with a minimum of 1 and asserts that the resolved `batch_size` evenly divides the rollout size so all minibatches have equal size.

Algo-specific config subclasses:
- `PPOConfig`: enforces `clip_range > 0`.
- `REINFORCEConfig`: validates `policy_targets` in {'returns','advantages'}.

### Environments (`utils/environment.py`, `gym_wrappers/*`)
- `build_env_from_config(config, **overrides)` expands a `Config` into the kwargs accepted by `build_env(...)` so call-sites can tweak seeds/rendering without duplicating plumbing.
- `build_env(env_id, n_envs, seed, subproc, obs_type, frame_stack, norm_obs, env_wrappers, env_kwargs, render_mode, record_video, record_video_kwargs)` builds a vectorized env:
  - ALE via `ale_py` with optional `obs_type in {'rgb','ram','objects'}` (objects uses OCAtari).
  - Optional base-env preprocessing (Gymnasium wrappers): `GrayScaleObservation` when `grayscale_obs` is set, and `ResizeObservation` when `resize_obs` is provided. `resize_obs` accepts `(width, height)`; `True` is treated as `(84, 84)` for convenience.
  - VizDoom via `gym_wrappers.vizdoom.VizDoomEnv` when `env_id` matches `VizDoom-*`.
  - Retro via `stable-retro` when `env_id` matches `Retro/*`.
  - Multi-armed bandits via `gym_envs.mab_env.MultiArmedBanditEnv` when `env_id` matches `Bandit-*` or `Bandit/<name>`.
  - Standard Gymnasium otherwise.
  - Applies registry wrappers from YAML: `EnvWrapperRegistry.apply` with `{ id: WrapperName, ... }` specs.
- Observation normalization: `normalize_obs` supports:
  - `false` (default): no normalization
  - `true` or `'rolling'`: SB3 `VecNormalize` running mean/std for observations
  - `'static'`: bounds-based `VecNormalizeStatic` (observations only)
  Reward normalization: enable with `normalize_reward: true` (SB3 `VecNormalize`). When `'static'` is selected for observations, reward normalization is ignored.
  - Optional `VecFrameStack` and `VecVideoRecorder`.
  - Each base env is wrapped with `EnvInfoWrapper`; the returned vec env is wrapped by `VecEnvInfoWrapper`, which exposes helpers like `.recorder(...)`, `get_return_threshold()`, `get_max_episode_steps()`, and `.is_rgb_env()` used throughout training/eval flows.
- Wrapper registry: `gym_wrappers.__init__` registers `PixelObservationWrapper` and domain wrappers like `PongV5_FeatureExtractor`, reward shapers, etc.

### Agents
- `BaseAgent` (LightningModule):
  - Creates `train`, `val`, and `test` envs with offset seeds and stage-specific video settings, then builds matching `RolloutCollector`s. Instantiates a `MetricsRecorder` (step key `train/cnt/total_vec_steps`), a `MetricsMonitor` used for alert hooks, and a `Run` object for run directory bookkeeping.
- Manual optimization in `training_step`; per-batch metrics are buffered in the recorder and flushed once per epoch by `trainer_callbacks.DispatchMetricsCallback`, which also updates history and forwards `train/*` / `val/*` snapshots to Lightning loggers. `MetricsMonitor.check()` surfaces alert messages (printed once per epoch), and `loggers.metrics_table_logger.PrintMetricsLogger` renders the console table using `config/metrics.yaml` precision/highlight rules.
  - Tracks best episode rewards via the collector: `train/roll/ep_rew/best` and `val/roll/ep_rew/best` are derived from the running best of `*/roll/ep_rew/mean` and appear in `metrics.csv`, W&B, and the terminal view.
  - Evaluation cadence honors `Config.eval_freq_epochs` and `eval_warmup_epochs`. `WarmupEvalCallback` disables validation until warmup completes, after which `validation_step` drives the collector-based evaluation + optional video capture described earlier.
  - Schedules: BaseAgent auto-wires any `*_schedule` fields in `Config` to `HyperparameterSchedulerCallback`s. Special setters are used when needed (e.g., `policy_lr` updates both the attribute and optimizer param groups; logged as `train/policy_lr`). Each schedule can now specify explicit `*_schedule_start_value` / `*_schedule_end_value` and `*_schedule_start` / `*_schedule_end`. Start/end positions accept either fractions (`0.25` → 25% of `max_timesteps`) or absolute vec steps (`>1`), and default to `[0, max_timesteps]` when omitted.
  - Logging: `WandbLogger` derives the project from `env_id`, `CsvLightningLogger` writes `runs/<id>/metrics.csv`, and `PrintMetricsLogger` keeps the terminal table in sync with Lightning's metric stream. Gradients are captured in W&B via `WandbLogger.watch(model, log='gradients', log_freq=100)`.
- `agents/ppo.PPO`:
  - Builds an actor-critic via `utils.policy_factory.build_policy_from_env_and_config`, computes PPO losses, and defaults to AdamW (configurable through `Config.optimizer`).
  - PPO losses consume mutable attrs like `clip_range`, `vf_coef`, and `ent_coef`, so schedules on these fields apply automatically.
  - Registers metric monitors for KL, clip fraction, and explained variance; alerts show up in the console when values leave recommended ranges.
  - Logs gradient norms for `actor_head`, `critic_head`, and `trunk` before clipping/optimizer step (exposed as `train/opt/grads/norm/*`).
- `agents/reinforce.REINFORCE`:
  - Builds a policy-only network via `utils.policy_factory.build_policy`, supports returns- or advantages-weighted policy targets (`config.policy_targets`), and disables GAE in the collector for classic REINFORCE behavior.
  - Logs diagnostics including entropy and PPO-style KL approximations comparing rollout vs current log-probs; Monte Carlo return mode is controlled by `Config.returns_type` (`mc:rtg` default, `mc:episode` to scale all timesteps equally).

### Rollouts and data pipeline (`utils/rollouts.py`, `utils/dataloaders.py`)
- `RolloutCollector` collects `n_steps` across `n_envs` and stores into a persistent CPU `RolloutBuffer` to minimize allocs.
- Supports GAE or Monte Carlo returns; handles `TimeLimit.truncated` with bootstrapped values; maintains rolling windows for ep stats and FPS.
- `flatten_slice_env_major` preserves image observations in CHW shape and only flattens vectors/scalars. Shapes returned:
  - Vectors/scalars: `(N*T, D)` or `(N*T, 1)`; Images (CHW): `(N*T, C, H, W)`.
- Terminal observations for time-limit truncations are coerced to the buffer's CHW shape before value bootstrapping.
- `build_index_collate_loader_from_collector` creates a `DataLoader` over indices with `MultiPassRandomSampler` to perform `n_epochs` passes over the same rollout without epoch resets.
- `evaluate_episodes(n_episodes, deterministic, timeout_seconds)` reuses the collector to gather finished episodes only, balances per-env episode targets, and updates running aggregates such as `ep_rew_best`, `ep_len_mean`, and both total environment frames (`total_timesteps`) and vector steps (`total_vec_steps`) consumed during evaluation.

### Models and policies (`utils/models.py`, `utils/policy_factory.py`)
- MLP actor-critic (`MLPActorCritic`) and policy-only (`MLPPolicy`) for discrete actions; distributions are `Categorical` from logits. `MLPPolicy` accepts either `input_dim`/`output_dim` or `input_shape`/`output_shape` for factory compatibility.
- CNN variants (`CNNActorCritic`, `CNNPolicy`) share a reusable `_CNNTrunk` (Conv2d stack + optional MLP). Models expect CHW inputs; a deterministic `_EnsureCHW` unflattens when given flat features (no HWC/CHW heuristics).
- `policy_factory` routes based on `policy` string or custom module classes and forwards `policy_kwargs`.

### Trainer and callbacks (`utils/trainer_factory.py`, `trainer_callbacks/*`)
- `build_trainer` constructs a PL `Trainer` with validation cadence controlled externally; progress bar/checkpointing disabled in favor of custom callbacks.
- Logging:
  - `WandbLogger`, `CsvLightningLogger` (writes `metrics.csv`), and `PrintMetricsLogger` (pretty console table). Lightning dispatches `log_dict` to all.
- Callbacks include:
  - `DispatchMetricsCallback`: resets per-epoch buffers, aggregates recorder metrics, and calls `pl_module.log_dict` / `MetricsRecorder.update_history` once per epoch for train/val.
  - `WarmupEvalCallback`: disables validation during warmup epochs and re-enables it with the configured cadence afterwards.
  - `ModelCheckpointCallback`: saves `epoch=XX.*` artifacts each eval epoch and refreshes `best.*` / `last.*` symlinks on fit end.
  - `VideoLoggerCallback`: watches `runs/<id>/videos` (train/val namespaces) and uploads new media files to W&B on epoch end.
  - `EarlyStoppingCallback`: supports stop conditions on total timesteps as well as train/eval reward thresholds (using env reward thresholds when requested).
  - `EndOfTrainingReportCallback`: writes `report.md` summarizing training at fit end.

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
- **Add an algorithm**: create `agents/<algo>.py` subclassing `BaseAgent`; implement `build_models`, `losses_for_batch`, `configure_optimizers`; wire dispatch in `agents/__init__.py::build_agent`.
- **Add an env wrapper**: implement under `gym_wrappers/<Name>/...` and register in `gym_wrappers/__init__.py` or via `EnvWrapperRegistry.register`.
- **Add config**: create a new block in `config/environments/*.yaml` with `inherits` and fields; prefer schedules like `lin_0.001`.
- **Metrics/logging**: adjust `config/metrics.yaml` / `utils.metrics_config` for precision/highlight rules, register additional `MetricsMonitor` callbacks when alerts are needed, and continue logging through `MetricsRecorder` buffers to keep the Lightning stream consistent.

### Conventions and gotchas
- Observations are flattened only for vectors/scalars; image observations are preserved as CHW tensors through the roll/DataLoader path. CNN models consume CHW directly (a small trunk handles shape consistency).
- Q-Learning assumes discrete obs/action spaces; do not auto-wrap discrete obs into vectors in `build_env` to avoid breaking tabular methods.
- When `eval_freq_epochs` is None or 0, validation is disabled. With warmup (`eval_warmup_epochs > 0`), all epochs up to and including the warmup boundary are skipped; evaluation resumes on the cadence grid (multiples of `eval_freq_epochs`) strictly after warmup. Example: warmup=50, freq=15 -> first eval at E=60 (epoch_idx=59).
- Use `config.max_timesteps` for progress-based schedules; ensure it’s set for linear decays to have effect.
- Training CLI uses `--config_id "<env>:<variant>"`; underscore-only IDs like `Env_Variant` are not parsed by `train.py`. Callers must supply an explicit variant when loading configs (either `<env>:<variant>` or `<env>`, `<variant>`), matching the `<project>_<variant>` keys produced when reading YAML.
- Env normalization: pass `normalize_obs: false|true|'rolling'|'static'` and optionally `normalize_reward: true` to enable SB3 normalization; defaults are off.

### Directory layout
```
agents/           # PPO, REINFORCE, base
loggers/          # custom Lightning/CSV/console loggers
utils/            # config, env, rollout, dataloaders, models, metrics, helpers
gym_wrappers/     # registry + wrappers (feature extractors, reward shaping, pixels)
trainer_callbacks/# metrics dispatch, warmup gating, checkpointing, early stopping, videos
gym_envs/         # custom environment implementations (e.g., MultiArmedBandit)
config/           # environment YAML configs
runs/             # run outputs: config.json, checkpoints/, videos/, metrics.csv, run.log
VIBES/            # agent helper docs (Architecture Guide, task playbooks)
```
