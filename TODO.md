# TODO

We currently use total_timesteps to tell us how many timesteps happened so far. And all our reporterd metrics are keyed under that metric. However, its not optimal
because I cant compare runs that have different n_envs. Since these envs are collected under same policy it doesnt seem fair when comparing to less envs. So I think
there should be another timestep metric that tracks the number of vectorized env.step() instead and that is the metric that everyone keys from. Think hard, make clean
changes.

## Pong-v5_objects

- Try frame stacking, may mitigate stickiness issues
- Check if env gets solved when its deterministic
- try debug server
- BUG: fix duplicate object error in feature extractor
- online feature normalization inside extractor (rolling averages)
- add env normalization support (use that instead of normalization from feature extractor)

## ALERTS

- add progress metric 0-1 that uses total_timesteps/max_timesteps when available
- FEAT: define min-max entropy on wandb dashboard
- BUG: alerts are not making into table logger
- FEAT: alert if action_mean doesnt start in expected bounds (initial policy not uniform)
- FEAT: alert if action_std doesnt start in expected bounds (initial policy not uniform)
- FEAT: search for known hard min/max bounds for metrics that are not currently in metrics.yaml and add them
- FEAT: infer min/max bounds based on configuration

## NEXT

- FEAT: generalize decay schedule further by specifying decay start
- FEAT: add ability to tell LLM to inspect last N runs by providing a run registry json that has timestamps and other metadata, always sort by timestamp descending, ensure lock on write
- FEAT: track approx_kl early stops in wandb dashboard
- FEAT: add support for scheduler min and progress decoupled from timesteps, perhaps by specifying percentage -> value tuples and have the scheduler interpolate between those, allowing dynamic linear schedules that perform differently across the training (eg: higher early, lower late)
- BUG: env smoke tests not passing
- FEAT: currently charts are indexed by n_envs * n_steps, index only by n_steps so we can compare performance of N_ENVS vs less envs; max_timesteps should also be indexed by that, as well as training progress
- FEAT: add support for agents to handle their own save/load logic
- FEAT: plot scaled losses together to understand their relative importance
- FEAT: track dead relus
- TEST: empirically verify that initial policy distribution is uniform (check if action mean starts at middle of action space and std is 0)
- BUG: bandit env crashes because it tries to record
- BUG: bandit training not running due to missing spec
- BUG: can't inspect bandit training

- BUG: training epoch is still running after validation early stop
- BUG: wandb run is being logged even when training is not started, bold terminal
- BUG: W&B workspace redundant print
- TEST: are sweeps still working?
- FEAT: track output distribution
- REFACTOR: dont let callbacks inject variables int o
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)
- TODO: learn how to read gradient graphs

## WISHLIST

- FEAT: add support to only start eval when ep_rew_mean crosses eval threshold
- FEAT: auto-tune n_steps to average steps between rewards
- FEAT: customize wandb dashboard to make action_mean min/max be computed from the action space defined in the spec, same for obs_mean
- TASK: solve Pong-v5_objects, get max reward
- TASK: solve Taxi-v3 with PPO, training stalls for unknown reasons
- FEAT: add support for run_play to run with random actions and/or user actions
- FEAT: add LunarLander-v3 randomization wrapper
- FEAT: add observation/action noise support
- FEAT: dispatch maintenance tasks with codex on cloud (night shift)
- FEAT: improve scheduling support (more generic, ability to define start/end and annealing schedule; NOTE: reset optimizer on learning rate changes?)
- FEAT: autotune n_envs (check tune_nenvs.md)
- FEAT: add LLM review support to inspector.py
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
- FEAT: add support for resuming training from a previous run (must restore optimizer, epoch, etc.; make restoring optimizer conditional, allow restarting with new config hyperparams)
- FEAT: add support for running sweep from existing run (using previous resume support)
- FEAT: allow downloading old runs from wandb when not available locally
- FEAT: zip and upload runs to wandb
- FEAT: add support for publishing run to Hugging Face Hub
- FEAT: add support for async eval (to avoid blocking training)
- FEAT: SEEDRL+PPO
