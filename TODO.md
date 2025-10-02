# TODO

# NEXT

- REFACTOR: ensure agents are themselves responsible for saving all of their artifacts to the run, in the same fashion it should also be possible to load back an agent by pointing to a run dir. NOTE: when saving/loading we need to make sure that we can get the agent back to the exact training state it was in when it was saved.
- inspect mc_returns and gae for unifinished episodes
- TEST: do highlighted rows also show alerts correctly?
- BUG: inspect not working because it cant retrieve action labels
- BUG: env smoke tests not passing
- FEAT: track dead relus
- BUG: training epoch is still running after validation early stop
- TEST: are sweeps still working?
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)
- TODO: learn how to read gradient graphs

- FEAT: Create MCP server that provides useful tools for claude code to be able to run training sessions and inspect training runs. This tool should have tools like the ability to list available environments and configs, list runs, start a run, etc.

## Pong-v5

- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Check best performance on deterministic env, if it reaches 20-21 then the feature extractor is probably ok (note: raise threshold for this env?)
- Try tanh and check if performance improves
- Confirm that we can get >=18 in stochastic env
- Confirm that Pong-v5 deterministic env reaches 20-21
- Check if we can get to 20-21 with RGB observations
- Frame stacking should work to turn POMDP into MDP
- TODO: add support for resuming with policy from previous run
- add env normalization support (use that instead of normalization from feature extractor)
- previous actions
- Search for where to check for SOTA scores on each env (ask gpt to research)

## MountainCar-v0

- FEAT: MountainCar-v0: rewardshaping; statecount bonus

## LunarLander-v3

- Solve baseline
- Increase difficulty using domain randomization
- Ensure eval uses domain randomization as well
- FEAT: add LunarLander-v3 randomization wrapper

## WISHLIST

- FEAT: add support to only start eval when ep_rew_mean crosses eval threshold
- FEAT: auto-tune n_steps to average steps between rewards
- TASK: solve Taxi-v3 with PPO, training stalls for unknown reasons
- FEAT: add support for run_play to run with random actions and/or user actions
- FEAT: add observation/action noise support
- FEAT: dispatch maintenance tasks with codex on cloud (night shift)
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
- FEAT: multidiscrete support
- FEAT: implement RND (Random Network Distillation)
- FEAT: MountainCar-v0: rewardshaping; statecount bonus
- FEAT: Recurrent PPO (eg: PPO-LSTM)