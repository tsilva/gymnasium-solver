# TODO

## NEXT

- TODO: cleanup pong feature extractor
- FEAT: center pong observations around 0, then after confirming superiority, remove feature and provide same featureset through normalization wrapper
- BUG: bandit training not running due to missing spec
- FEAT: add support for scheduler min
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
