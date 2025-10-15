# TODO

## Remote training loop

- Ensure training is easy to launch (merge with train.py)
- Ensure ROMs are imported remotely
- Is it possible to store built containers?

# NEXT

- TODO: ensure clip_range_vf exists
- add support for masking invalid action combos, making sure that the highest logit wins (the loser is masked)
- BUG: run_play.py user input not working for Atari envs
- TEST: CleanRL's envpool implementation
- TEST: is last eval in uploaded zip file
- FEAT: add support for adding shared configs between envs (eg: atari defaults, vizdoom defaults)
- FEAT: single plot with fraction of scaled losses
- TEST: ensure all checkpoints get stored in wandb (check storage limits)
- FEAT: run_inspect.py add support for monitoring rollouts with different hyperparams
- FEAT: train from previous run, use that to do multiple runs until convergence (new run loads param); should we restore optimizer? confirm new run starts with same performance, check if it evolves better with same or dfiferent optimizer
- CHECK: should we reset optimizer when we use another learning rate
- TEST: ensure evaluation is ran exactly same way as train (eg: alevecenv)
- TODO: make key capture not require enter, allow h to show all shortcuts
- TEST: logged to correct projects
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)
- TODO: add env normalization support, make sure normalization is saved
- TODO: add action number/ label before each frame stack image in inspect (allows easily seeing which action was performed when each frame was seen)
- FEAT: CNN filter/activation inspectors
- BUG: is eval not imposing timelimits?

## SuperMarioBros-Nes

- https://github.com/nemanja-m/super-mario-agent - implement architecture; implement stochastic frame skip wrapper
- don't use sticky actions wrapper during eval
- TODO: confirm that sticky actions wrapper guarantees that policy performs those actions after training
- TODO: use action combo instead of multibinary
- EpisodicLifeEnv wrapper
- Penalize time spent more?
- TEST: is eval not imposing timelimits?
- REWARD: instead of measuring velocity can we just use time passed to deduct movement reward?
- REWARD: abort when no reward increased for N steps?
- TODO: try with variable frame skip including no frame skip (eg: starting in 1)
- Should we remove frame skip in evaluation?


## VizDoom-v0

- FEAT: run_play.py make random mode instead of random sampling just do regular init but with random policy (this way action masking will still be applied)
- BUG: E1M1 cant open doors
- TEST: ensure rewardshaper is working (test manually)
- FEAT: reduce obs for vizdoom
- FEAT: read vizdoom vars as input
- FEAT: action masking to block impossible combos (eg: left+right)
- episodes not incrementing in user mode
- BUG: ensure vizdoom reward shaper is truly working by debugging with manual control
- FEAT: reward shape defend the center to penalize shooting
- TEST: is it faster to learn defendtheX after basic env
- FEAT: add support for regulating difficulty (doom_skill)
- TEST: are seeds working?
- TEST: is frameskip working
- TEST: is frameskip useful?
- BUG: action labels are invalid (use keyboard input to figure out correct labels)
- LEARN: [Medium Lesson](https://lkieliger.medium.com/playing-doom-with-deep-reinforcement-learning-part-3-boosting-performance-with-reward-shaping-b1de246bda1d
https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
- LEARN: [HF Course Lesson](https://huggingface.co/learn/deep-rl-course/unit8/hands-on-sf)
- [Direct Future Prediction](https://flyyufelix.github.io/2017/11/17/direct-future-prediction.html)
- LEARN: [RoE - Rarity of Events](https://arxiv.org/pdf/1803.07131)
- imitation learning (unify pretraining process with minari support)
- LEARN: [Actor-Mimic](https://arxiv.org/abs/1511.06342) - distil expert policies into single model then finetune it on deathmatch
- WISH: ability to play against AI agent
- CURRICULUM:
VizDoom-MyWayHome-v0
VizDoom-HealthGathering-v0
VizDoom-DefendTheLine-v0
VizDoom-DefendTheCenter-v0
VizDoom-Basic-v0
VizDoom-PredictPosition-v0
VizDoom-TakeCover-v0
VizDoom-DeadlyCorridor-v0
VizDoom-Deathmatch-v0


## Pong-v5

- NOOPs are overconfident, try training without NOOPs (may change training dynamics)
- TODO: learn how to read gradient graphs
- Check best performance on deterministic env, if it reaches 20-21 then the feature extractor is probably ok (note: raise threshold for this env?)
- Confirm that Pong-v5 deterministic env reaches 20-21
- Check if we can get to 20-21 with RGB observations
- Frame stacking should work to turn POMDP into MDP
- TODO: add support for resuming with policy from previous run
- add env normalization support (use that instead of normalization from feature extractor)
- Search for where to check for SOTA scores on each env (ask gpt to research)

## LunarLander-v3

- Solve baseline
- Increase difficulty using domain randomization
- Ensure eval uses domain randomization as well
- FEAT: add LunarLander-v3 randomization wrapper
- FEAT: reward shape lunarlander to train faster by penalizing long episodes

## Taxi-v3

- TASK: solve Taxi-v3 with PPO, training stalls for unknown reasons

## WISHLIST

- FEAT: batch norm support
- FEAT: layer norm support
- FEAT: add observation/action noise support
- FEAT: add LLM review support to inspector.py
- FEAT: add support for continuous environments (eg: LunarLanderContinuous-v2)
- FEAT: add A2C support
- FEAT: add [Minari](https://minari.farama.org/) support
- FEAT: add support for image environment training (eg: CNNs + preprocessing + atari preprocessing)
- FEAT: agent hyperparam autotuner
- FEAT: add support for dynamics models (first just train and monitor them, then leverage them for planning)
- FEAT: support for multi-env rollout collectors (eg: solve multiple envs at once; eg: train on multiple configs of LunarLander-v3)
- FEAT: add support for curriculum learning (take same model through N challenges sequentially)
- FEAT: add support for publishing run to Hugging Face Hub
- FEAT: SEEDRL+PPO
- FEAT: multidiscrete support
- FEAT: implement RND (Random Network Distillation)
- FEAT: Recurrent PPO (eg: PPO-LSTM)
- FEAT: ensure huggingface uploader, publishes run URL and 
