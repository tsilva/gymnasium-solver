# TODO

## NEXT

- BUG: videos not being recorded to run
- TEST: is wandb reporting same metrics as table?
- TEST: is table showing correct metrics?
- BUG: checkpoint jsons not storing correct metrics
- BUG: wandb run is being logged even when training is not started, bold terminal
- TODO: optimal single metric for best training (samples/reward)
- TODO: dont tie eval_freq_epochs and eval_warmup_epochs to epochs because ppos updates for many epochs; consider changing eval_freq to timesteps so it doesnt depend on n_updates
- FEAT: prefix metrics (eg: time/epoch, time/step, etc., env/obs_mean, env/obs_std, env/)
- FEAT: make boot and shutdown times faster
- FEAT: wandb bug may caused by lack of .finish()
- BUG: can't inspect bandit training
- BUG: W&B workspace redundant print
- TEST: are sweeps still working?
- FEAT: track output distribution
- REFACTOR: consolidate single-trajectory return/advantage helpers by moving run_inspect.py:compute_mc_returns and run_inspect.py:compute_gae_advantages_and_returns onto shared utilities next to utils/rollouts.py:compute_batched_mc_returns/compute_batched_gae_advantages_and_returns; acceptance: inspector outputs stay the same and rollout-related tests remain green.
- REFACTOR: deduplicate YAML config discovery by reusing Config loader logic between scripts/smoke_test_envs.py:_collect_env_config_map/_resolve_inheritance and utils/config.py:Config.build_from_yaml via a shared utility (e.g., utils/config_loader.collect_raw_configs); acceptance: both callers produce identical config maps without manual field lists and existing config-selection tests still pass.
- REFACTOR: unify run directory resolution and @latest-run handling by replacing run_publish.py:resolve_run_dir symlink traversal with utils/run.py:Run.from_id and utils/run.py:list_run_ids; keep the "pick best run when --run-id is omitted" heuristic in run_publish but use Run/_resolve_run_dir for @latest-run and constants from utils/run.py. Acceptance: default run selection remains identical and tests using utils.run (e.g., tests/test_run_manager.py) stay green.
- REFACTOR: reuse checkpoint name constants across modules. Replace hardcoded lists in run_publish.py:extract_run_metadata with utils/run.py:BEST_CKPT_NAMES and LAST_CKPT_NAMES and/or utils/run.py:Run.checkpoint_choices(); acceptance: representative checkpoint selection matches current behavior and tests/tests_checkpoint.py remain green.
- REFACTOR: centralize checkpoint loading/metadata parsing. Introduce utils/checkpoint.py:extract_metrics(ckpt_path) used by run_publish.py:extract_run_metadata and utils/policy_factory.py:load_policy_model_from_checkpoint to avoid duplicate torch.load logic and field probing; acceptance: metadata fields (epoch, total_timesteps, best/current eval reward) are unchanged while error handling is consistent.
- REFACTOR: centralize best-video discovery. Move run_publish.py:_find_videos_for_run/_find_best_video_for_run into a shared helper (e.g., utils/run.py:Run.best_video_path or utils/videos.py) and reuse from trainer_callbacks/end_of_training_report.py and run_publish.py; acceptance: the selected preview video remains the same and tests/test_best_video_symlink.py still pass.
- REFACTOR: extract policy update diagnostics shared by PPO/REINFORCE. Create utils/rl_diagnostics.py:compute_policy_update_diagnostics(old_logprobs, new_logprobs, clip_range) returning {clip_fraction, kl_div, approx_kl}; replace duplicated calculations in agents/ppo/ppo_agent.py:losses_for_batch and agents/reinforce/reinforce_agent.py:losses_for_batch. Acceptance: logged metrics are numerically identical and PPO KL early-stop triggers at the same epochs.
- REFACTOR: extract batch normalization helpers. Add utils/tensors.py:normalize_batch(x, eps=1e-8) and use in agents/ppo/ppo_agent.py and agents/reinforce/reinforce_agent.py instead of inline (x - mean)/(std + 1e-8); acceptance: training metrics are unchanged when normalize_advantages/returns == "batch".
- REFACTOR: extend YAML config discovery dedup to run_publish. Update run_publish.py:_guess_config_id_from_environments to reuse the shared loader proposed in utils/config_loader (see related REFACTOR above) rather than reimplementing file scanning; acceptance: inferred config_id remains the same for existing envs/variants.
- REFACTOR: standardize latest-run constants. Replace literal "@latest-run"/"latest-run" checks in run_publish.py and related scripts with utils/run.py:LATEST_SYMLINK_NAMES; acceptance: no behavior change, but sources of truth are centralized.
- REFACTOR: centralize symlink updates. Replace manual unlink/symlink logic in utils/run.py:_set_latest_symlink with utils/filesystem.py:update_symlink for portability and consistency with trainer_callbacks/model_checkpoint.py; acceptance: latest-run symlink behavior is identical across platforms and tests/test_run_manager.py remains green.
- REFACTOR: agents/base_agent.py:validation_step and agents/base_agent.py:on_fit_end — extract evaluate-and-record logic into a shared helper (e.g., utils/eval.py:evaluate_and_record(run, env, collector, video_path, episodes, deterministic)); acceptance: same videos and sidecar JSON produced; risks: coupling to VecEnv recorder API; size: M.
- REFACTOR: trainer_callbacks/early_stopping.py:EarlyStoppingCallback.on_train_epoch_end — replace direct write to pl_module._early_stop_reason with a public method on BaseAgent (e.g., agents/base_agent.py:set_early_stop_reason(reason)); acceptance: identical reason string in reports; risks: add method and update callback; size: S.
- REFACTOR: utils/dataloaders.py:build_index_collate_loader_from_collector — reduce signature width by introducing an options dataclass (e.g., IndexCollateLoaderOptions) for rarely-changed kwargs; acceptance: DataLoader behavior identical; risks: touch BaseAgent.train_dataloader call-site; size: M.
- REFACTOR: gym_wrappers/env_video_recorder.py:EnvVideoRecorder.__init__ — encapsulate overlay parameters into a VideoOverlayStyle dataclass; acceptance: default overlay identical; risks: adjust construction in utils/environment.py only; size: S.
- REFACTOR: agents/base_agent.py:_build_trainer_loggers__* and _build_trainer_loggers — move logger assembly to utils/logging_factory.build_trainer_loggers(agent) to hide plumbing; acceptance: same loggers and step metric configuration; risks: preserve wandb init semantics; size: M.
- REFACTOR: agents/base_agent.py:_update_schedules__policy_lr and agents/ppo/ppo_agent.py:_update_schedules__clip_range — centralize scheduling math/logging in utils/schedulers (e.g., linear(value0, progress)); acceptance: logged values unchanged; risks: order-of-operations drift; size: S/M.
- REFACTOR: run_inspect.py:run_episode helpers (_ensure_hwc, _split_stack, _make_grid, _vector_stack_to_frames) — move to utils/visualization.py (or utils/inspect_helpers.py) and import; acceptance: identical images/behavior; risks: import cycles; size: M.
- REFACTOR: run_inspect.py:build_ui — decompose into inspector/ui_components.py (controls, handlers, timers) with a thin coordinator; acceptance: UI behavior and controls unchanged; risks: gradio version quirks; size: L.
- REFACTOR: run_publish.py:[resolve_run_dir, _find_videos_for_run, _find_best_video_for_run, extract_run_metadata, build_model_card] — extract discovery and card-building into utils/run_artifacts.py; acceptance: same repo contents and README; risks: HF Hub API surface; size: L.
- REFACTOR: trainer_callbacks/wandb_video_logger.py:WandbVideoLoggerCallback._process — split into smaller helpers (scan_pending, filter_by_min_age, group_by_key, to_wandb_media, log_group) and share step resolution via utils/metrics_config.metrics_config; acceptance: same media logged at same steps; risks: race conditions; size: M.
- REFACTOR: utils/config.py:Config.validate — split into per-domain validators (training/eval/env/algo) to reduce branches and enable reuse; acceptance: same validation results/messages; risks: slight error-message wording changes; size: M.
- REFACTOR: agents/base_agent.py:train_dataloader/on_train_epoch_start — add ensure_rollout_for_current_epoch() to own first-epoch bootstrap vs per-epoch collect; acceptance: no double-collect; metrics match; risks: off-by-one on epoch 0; size: S.
- REFACTOR: utils/environment.py:build_env — extract render/video handshake into a helper (e.g., utils/video_recording.ensure_rgb_render_support(env, render_mode)); acceptance: same warnings and behavior; risks: wrapper-order assumptions; size: M.

## WISHLIST

- FEAT: dispatch maintenance tasks with codex on cloud (night shift)
- FEAT: improve scheduling support (more generic, ability to define start/end and annealing schedule; NOTE: reset optimizer on learning rate changes?)
- FEAT: add env normalization support 
- FEAT: autotune n_envs (check tune_nenvs.md)
- FEAT: add LLM review support to inspector.py
- TASK: solve Pong-v5_objects, get max reward
- TASK: solve Taxi-v3 with PPO, training stalls for unknown reasons
- FEAT: add support for continuous environments (eg: LunarLanderContinuous-v2)
- FEAT: add A2C support
- FEAT: add [Minari](https://minari.farama.org/) support
- FEAT: add support for image environment training (eg: CNNs + preprocessing + atari preprocessing)
- FEAT: agent hyperparam autotuner (eg: with codex-cli)
- FEAT: reward shape lunarlander to train faster by penalizing long episodes
- FEAT: CartPole-v1, create reward shaper that prioritizes centering the pole
- FEAT: add support for dynamics models (first just train and monitor them, then leverage them for planning)
- FEAT: support for multi-env rollout collectors (eg: solve multiple envs at once)
- FEAT: add support for curriculum learning (take same model through N challenges sequentially)
- FEAT: add support for resuming training from a previous run (must restore optimizer, epoch, etc.)
- FEAT: add support for publishing run to Hugging Face Hub
- FEAT: add support for async eval (to avoid blocking training)
- FEAT: SEEDRL+PPO
