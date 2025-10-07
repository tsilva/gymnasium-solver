# TODO

# NEXT

- dynamic sweep
- add support for running training in beefed up CPU machine in modal
- consider intelligent eval strategy
- parallelize sweeps (docker, spot instances, ray?)
- FEAT: train from previous run, use that to do multiple runs until convergence (new run loads param); should we restore optimizer? confirm new run starts with same performance, check if it evolves better with same or dfiferent optimizer
- CHECK: should we reset optimizer when we use another learning rate
- BUG: rgb env not working
- FEAT: add max episode steps support (cartpole, atari, vizdoom, nes)
- TEST: is last eval in uploaded zip file
- TEST: ensure evaluation is ran exactly same way as train (eg: alevecenv)
- TEST: ensure frameskip+max is being applied to vizdoom and retro
- TODO: make key capture not require enter, allow h to show all shortcuts
- BUG: vecobs not showing action labels for pong rgb
- TEST: vizdoom env works (with run_play and run_inspect)
- TEST: super mario env works (with run_play and run_inspect)
- TEST: smoke tests pass
- FEAT: speed up eval as much as possible (async mode with few changes)
- TODO: trace hyperparam tuning process and adjust
- TEST: logged to correct projects
- TEST: tune with agent
- TEST: are sweeps still working?
- TODO: remaining codebase TODOs (eg: lots of AI slop to refactor)
- TODO: add env normalization support, make sure normalization is saved
- TODO: add fit to container zoom in inspect
- TODO: add action number/ label before each frame stack image in inspect (allows easily seeing which action was performed when each frame was seen)

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
- FEAT: add support for async eval (to avoid blocking training)
- FEAT: SEEDRL+PPO
- FEAT: multidiscrete support
- FEAT: implement RND (Random Network Distillation)
- FEAT: Recurrent PPO (eg: PPO-LSTM)