# TODO

- BUG: Cartpole-v1 eval not working (record training and or eval videos)
# MODAL

- TASK: run sweep remotely
- TASK: let agent/script say which run performs best (best ep_rew_mean slope from 0 to last step)
- BUG: cant seem to detach modal runs
- BUG: runs executed through modal dont seem to have same wandb run id
- BUG: clean exit run_play on ctrl+c
- BUG: remote runs get killed if I close terminal?

# SWEEPS

- FEAT: add sweep.py with --backend modal support
- FEAT: sweep from run/checkpoint/

# RUN_PLAY

- step by step support
- run play without ui
- TASK: dont use alevecenv for evaluation, at least not in runplay, consider just running single env directly
- Add fps changer
- Add window layout snapshot button
- BUG: run_play.py user input not working for Atari envs

# NEXT

- BUG: why cant I resume run from exactly same perf location
- BUG: cant train VizDoom-Basic-v0 when frame_stack is 1
- FEAT: run_inspect.py add support for monitoring rollouts with different hyperparams
- THINK: how to audit reward structure / mean, std, outliers, and histogram
- TEST: ensure evaluation is ran exactly same way as train (eg: alevecenv)
- TODO: add env normalization support, make sure normalization is saved
- TODO: add action number/ label before each frame stack image in inspect (allows easily seeing which action was performed when each frame was seen)
- BUG: is eval not imposing timelimits?
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)

## SuperMarioBros-Nes

- FEAT: wrapper that patches parts of OBS (eg: hud)
- GOAL: ensure level-1-1 can be completed with ~100% win rate
- EXP: train on level 2 after mastering level 1, check if it forgets level 1
- FEAT: stack action just as frames 
- FEAT: resize to preserve aspect ratio (to avoid distortions)
- FEAT: crop mario frames
- TEST: quantize image (before or after downscaling)
- TEST: instead of cropping HUDS just mask them with black so that we can keep a constant image size between envs
- If VF stabilized try increasing LR
- mario: make policy more robust to timing differences by finetuning with variable frameskip
- TODO: Huber Loss for VF; value normalization/PopArt
- TEST: how mario behaves if reward is just curiosity
- LEARN: reward standardization - Use automatic weighting (e.g., Pop-Art normalization or reward standardization) to balance magnitudes
- TEST: try lowering vf_coef to stabilize VF clipping
- TUNE: can I train faster with more n_envs? (launch multiple and check FPS plots)
- FEAT: figure out how to consider training finished when level changes
- FEAT: train next levels using starting point of previous levels, create master checkpoint for each level
- FEAT: distil policy that plays all levels
- FEAT: try training directly on a different level per env?
- https://github.com/nemanja-m/super-mario-agent - implement architecture; implement stochastic frame skip wrapper
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
- TEST: are we using frameskip? should we?
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

- FEAT: dropout
- FEAT: merge gymnasium recorder repo with this one
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
- FEAT: implement RND (Random Network Distillation)
- FEAT: Recurrent PPO (eg: PPO-LSTM)
- FEAT: ensure huggingface uploader, publishes run URL and 
- GoExplore: downsample images and threshold them
use parallelization to init eqch from a different archive state, sample based on inverse visit count
after rollout ia collected, traverse it to update archive with new cells and trajectories
do n turns of this loop
then train for n epochs om archive nor on rollout
this way we unify both