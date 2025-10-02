# TODO

# NEXT

- TASK: run stable-retro compile.sh
- TEST: logged to correct projects
- TEST: tune with agent
- TEST: are sweeps still working?
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)

## Pong-v5

- TODO: learn how to read gradient graphs
- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Check best performance on deterministic env, if it reaches 20-21 then the feature extractor is probably ok (note: raise threshold for this env?)
- Try tanh and check if performance improves
- Confirm that we can get >=18 in stochastic env
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

- FEAT: autotune n_envs (check tune_nenvs.md)
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