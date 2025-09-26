# TODO

# NEXT

- BUG: wrong project id logged in wandb
- BUG: spec is being retrived from env config
- BUG: inspect not working because it cant retrieve action labels
- BUG: investigate mc_return in pong, seems busted
- key eval freq from vec steps
- BUG: fps seems slow
- NEXT: only show certain metrics in table logger
- Add alert for when ep_rew_mean starts stalling / downward trend
- Make sure metrics.yaml is up to date with all metrics with correct names
- FEAT: add support for scheduler min and progress decoupled from timesteps, perhaps by specifying percentage -> value tuples and have the scheduler interpolate between those, allowing dynamic linear schedules that perform differently across the training (eg: higher early, lower late); generalize decay schedule further by specifying decay start
- PERF: dont fork subprocesses before confirming run
- FEAT: add ability to tell LLM to inspect last N runs by providing a run registry json that has timestamps and other metadata, always sort by timestamp descending, ensure lock on write
- BUG: env smoke tests not passing
- FEAT: add support for agents to handle their own save/load logic
- FEAT: track dead relus
- TEST: empirically verify that initial policy distribution is uniform (check if action mean starts at middle of action space and std is 0)
- BUG: bandit env crashes because it tries to record
- BUG: bandit training not running due to missing spec
- BUG: can't inspect bandit training
- BUG: training epoch is still running after validation early stop
- BUG: wandb run is being logged even when training is not started, bold terminal
- TEST: are sweeps still working?
- FEAT: track output distribution
- REFACTOR: dont let callbacks inject variables int o
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)
- TODO: learn how to read gradient graphs

## Pong-v5_objects

- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- TODO: rerun with these hyperparams https://wandb.ai/tsilva/ALE-Pong-v5_objects/runs/jfr90820/overview
- Git bisect and find cause of deterministic not converging
- Frame stacking should work to turn POMDP into MDP
- TODO: add support for resuming with policy from previous run
- try debug server
- BUG: fix duplicate object error in feature extractor (run step by step in human mode and compare conflicted object data, against frame, create script for this)
- add env normalization support (use that instead of normalization from feature extractor)
- previous actions
- Search for where to check for SOTA scores on each env (ask gpt to research)

## LunarLander-v3

- Solve baseline
- Increase difficulty using domain randomization
- Ensure eval uses domain randomization as well

## WISHLIST

- FEAT: add support to only start eval when ep_rew_mean crosses eval threshold
- FEAT: auto-tune n_steps to average steps between rewards
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
- FEAT: P3