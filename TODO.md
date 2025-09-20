# TODO

## NEXT

- BUG: envinfo not working for pong-v5-objects
- BUG: alerts are not making into table logger
- BUG: training epoch is still running after validation early stop
- BUG: wandb run is being logged even when training is not started, bold terminal
- FEAT: make boot and shutdown times faster
- FEAT: prefix metrics (eg: time/epoch, time/step, etc., env/obs_mean, env/obs_std, env/)
- BUG: can't inspect bandit training
- BUG: W&B workspace redundant print
- TEST: are sweeps still working?
- FEAT: track output distribution
- REFACTOR: dont let callbacks inject variables int o
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)

## WISHLIST

- FEAT: add LunarLander-v3 randomization wrapper
- FEAT: add env normalization support 
- FEAT: add observation/action noise support
- FEAT: dispatch maintenance tasks with codex on cloud (night shift)
- FEAT: improve scheduling support (more generic, ability to define start/end and annealing schedule; NOTE: reset optimizer on learning rate changes?)
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
- FEAT: support for multi-env rollout collectors (eg: solve multiple envs at once; eg: train on multiple configs of LunarLander-v3)
- FEAT: add support for curriculum learning (take same model through N challenges sequentially)
- FEAT: add support for resuming training from a previous run (must restore optimizer, epoch, etc.)
- FEAT: add support for publishing run to Hugging Face Hub
- FEAT: add support for async eval (to avoid blocking training)
- FEAT: SEEDRL+PPO

## Task report: cleanup_docstrings (2025-09-20)

- Compressed verbose multi-line docstrings to one-liners across key modules: `utils/logging.py`, `utils/formatting.py`, `utils/torch.py`, `utils/models.py`, `utils/policy_ops.py`, `utils/metrics_serialization.py`, `utils/metrics_config.py`, `utils/metrics_recorder.py`, `utils/rollouts.py`, `trainer_callbacks/early_stopping.py`, `trainer_callbacks/hyperparameter_scheduler.py`, `run_inspect.py`, `run_publish.py`, and scripts under `scripts/`.
- Removed example/usage blocks and long prose; kept intent-focused summaries only.
- No `Args/Parameters/Returns/Raises/Examples` sections remain (grep clean).
- Follow-ups: a few large files still have many helpers with terse docstrings; further trimming could remove trivial restatements if desired, but current pass keeps clarity without noise.

## Task report: cleanup_dead_code (2025-09-20)

- Removed a dead debug line in `scripts/strip_yaml_defaults.py` (an `if False` no-op dump). Root cause: leftover debug scaffold not executed; safe deletion reduces noise.
- Pruned commented-out early-return lines in `trainer_callbacks/dispatch_metrics.py` that were known to break `val` stage. Root cause: stale guard kept as commented code; minimal removal keeps intent without misleading dead code.
- Vulture not available (`uv run vulture .` missing). Used `scripts/find_unused_defs.py` + `rg` to audit. Many candidates were false positives due to dynamic hooks (Lightning callbacks, watchdog FS handlers, metric bundles). Left those intact.
- No functional behavior changes; deferred broader refactors to focused tasks.
