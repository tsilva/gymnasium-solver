# TODO

## NEXT

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