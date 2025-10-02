# TODO

# NEXT

- TEST: all running
- TEST: logged to correct projects
- TEST: run registry
- TEST: videos recorded?
- BUG: env smoke tests not passing
- BUG: why no action historgram for cartpole?
- BUG: training epoch is still running after validation early stop

- TEST: do highlighted rows also show alerts correctly?
- BUG: inspect not working because it cant retrieve action labels
- TEST: are sweeps still working?
- TODO: learn how to read gradient graphs
- FEAT: Create MCP server that provides useful tools for claude code to be able to run training sessions and inspect training runs. This tool should have tools like the ability to list available environments and configs, list runs, start a run, etc. Ask agent to figure out exactly which tools would be optimal for it to be easily launch, stop and inspect training runs then add them.
- FEAT: zip and upload runs to wandb
- FEAT: add support for run_play to run with random actions and/or user actions
- FEAT: add support for running sweep from existing run (using previous resume support)
- FEAT: allow downloading old runs from wandb when not available locally
- FEAT: add support to only start eval when ep_rew_mean crosses eval threshold (or at fraction of)
- REFACTOR: simplify run_inspect.py code
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)

## REFACTOR

1. utils/rollouts.py (1,067 lines)

Why: Massively complex single file with
multiple concerns
- RolloutCollector class alone is 683 lines
- Mixes multiple classes: RolloutTrajectory,
RollingWindow, RunningStats, RolloutBuffer,
RolloutCollector
- Complex return/advantage computation logic
intertwined with rollout collection
- Episode tracking, metrics aggregation,
evaluation logic all in one class

Opportunities:
- Split into separate files:
rollout_buffer.py, rollout_collector.py,
rollout_stats.py, returns_advantages.py
- Extract episode processing logic from
RolloutCollector
- Separate metrics computation from
collection logic
- Move utility functions (lines 12-175) to
dedicated module

---
2. agents/base_agent.py (690 lines)

Why: God class with too many
responsibilities
- Complex callback building with inline
schedule resolution (lines 416-566, 150 
lines)
- Logger building scattered across 3 methods
(lines 353-414)
- Hyperparameter management mixed with
training logic (lines 629-676)
- Environment/rollout collector building
could be extracted

Opportunities:
- Extract CallbackBuilder class for callback
construction
- Extract ScheduleResolver helper for
schedule configuration (lines 452-522)
- Move logger building to
utils/trainer_loggers.py
- Extract hyperparameter management to
separate mixin/class

---
3. utils/config.py (649 lines)
---
4. utils/train_launcher.py (369 lines)

Why: Multiple unrelated concerns in one
module
- Config merging logic (W&B, debugger)
(lines 22-82)
- Pre-fit summary building (lines 132-193,
62 lines)
- Environment listing with fuzzy matching
(lines 261-369, 109 lines)

Opportunities:
- Extract summary building to
utils/training_summary.py
- Move environment listing to
utils/environment_registry.py
- Keep only core launch logic in
train_launcher.py

---
🟡 MEDIUM PRIORITY

5. utils/environment.py (298 lines)

Why: Long function with duplication between
vectorization paths
- build_env function is 201 lines (lines
91-291)
- Two major conditional branches: ALE native
(lines 147-198) vs standard (lines 199-264)
- Duplication in wrapper application,
seeding, video recording setup
- Multiple _build_env_* builders with
similar structure (lines 16-87)

Opportunities:
- Extract vectorization paths into separate
builder functions
- Create env builder registry/strategy
pattern for different env types
- Unify common setup logic (seeding,
wrappers, video recording)

---
Summary Stats:

- 5 files identified for simplification
- Total lines: 3,073 lines across these
files
- Estimated reduction potential: 30-40%
through encapsulation and extraction
- Primary patterns: God classes, mixed
concerns, long methods, duplication

Would you like me to deep-dive into any
specific file to create a detailed
refactoring plan?

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