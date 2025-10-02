# TODO

# NEXT

- TEST: logged to correct projects
- BUG: training epoch is still running after validation early stop
- run_play.py show action histogram
- make sure all smoke tests are passing
- TEST: are sweeps still working?
- TODO: learn how to read gradient graphs
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)

## REFACTOR

1. utils/rollouts.py (1,067 lines)

Why: Massively complex single file with multiple concerns
- RolloutCollector class alone is 683 lines
- Mixes multiple classes: RolloutTrajectory, RollingWindow, RunningStats, RolloutBuffer, RolloutCollector
- Complex return/advantage computation logic intertwined with rollout collection
- Episode tracking, metrics aggregation, evaluation logic all in one class

Opportunities:
- Split into separate files: rollout_buffer.py, rollout_collector.py, rollout_stats.py, returns_advantages.py
- Extract episode processing logic from RolloutCollector
- Separate metrics computation from collection logic
- Move utility functions (lines 12-175) to dedicated module

---

2. agents/base_agent.py (690 lines)

Why: God class with too many responsibilities
- Complex callback building with inline schedule resolution (lines 416-566, 150  lines)
- Logger building scattered across 3 methods (lines 353-414)
- Hyperparameter management mixed with training logic (lines 629-676)
- Environment/rollout collector building could be extracted

Opportunities:
- Extract CallbackBuilder class for callback construction
- Extract ScheduleResolver helper for schedule configuration (lines 452-522)
- Move logger building to utils/trainer_loggers.py
- Extract hyperparameter management to separate mixin/class

---

4. utils/train_launcher.py (369 lines)

Why: Multiple unrelated concerns in one module
- Config merging logic (W&B, debugger) (lines 22-82)
- Pre-fit summary building (lines 132-193, 62 lines)
- Environment listing with fuzzy matching (lines 261-369, 109 lines)

Opportunities:
- Extract summary building to utils/training_summary.py
- Move environment listing to utils/environment_registry.py
- Keep only core launch logic in train_launcher.py

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

## CartPole-v1

- FEAT: CartPole-v1, create reward shaper that prioritizes centering the pole

## MountainCar-v0

- FEAT: MountainCar-v0: rewardshaping; statecount bonus

## LunarLander-v3

- Solve baseline
- Increase difficulty using domain randomization
- Ensure eval uses domain randomization as well
- FEAT: add LunarLander-v3 randomization wrapper
- FEAT: reward shape lunarlander to train faster by penalizing long episodes

## WISHLIST

- FEAT: autotune n_envs (check tune_nenvs.md)
- TASK: solve Taxi-v3 with PPO, training stalls for unknown reasons
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
- FEAT: add support for async eval (to avoid blocking training)
- FEAT: SEEDRL+PPO
- FEAT: multidiscrete support
- FEAT: implement RND (Random Network Distillation)
- FEAT: Recurrent PPO (eg: PPO-LSTM)