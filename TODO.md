- FEAT: hp vs pf metrics
- FEAT: add discrete env support (solve taxi-v3, embeddings probably not sized correctly)

- FEAT: add normalization support 
- FEAT: add sweeping support
  
  Optionally:
  
  - Add - { id: PongV5_RewardShaper } to env_wrappers for a shaped variant.
  
  Would you like me to:
  
  - Patch the YAML to add ppo_tuned2 with the above, and set up a quick run?
  - Also fix the get_reward_treshold typo (safe, unrelated to learning)?
  
  If you prefer minimal change first, I’ll only bump clip_range to 0.2 and policy_lr to 5e-4 and rerun to verify approx_kl moves into the
  0.01–0.02 band.


- TODO: how is pongv5 determinism enforce
- TODO: run pongv5 sweep
- REFACTOR: simplify CNN policy creation, test manually first
- BUG: restore grayscaling / resizing logic; think how to unify with atari preprocessing (probably just inspect internals and extract)
- FEAT: add kl early stopping to PPO
- BUG: fix cnn policy training, in separate file create env and inspect traversing layers
- FEAT: overridable rollout collector factory methods instead of extra parameter method
- FEAT: log checkpoint times to wandb
- BUG: reset optimizer on learning rate changes
- TODO: make sure max timelimit is logged at training start
- TODO: beat lunarlander with ppo and only then with reinforce (trun[cate episode lengths for faster training)
- REFACTOR: consider changing eval_freq to timesteps so it doesnt depend on n_updates
- FEAT: figure out if I should log variance instead of std
- TODO: x metric cant be total timesteps because that increases with parallelization.... must be n_updates? index by epoch? or add updates metric?
- TODO: optimal single metric for best training (samples/reward)
- TODO: add kl divergence metric to reinforce
- TODO: keep working on guide
- REFACTOR: move hidden_dims inside policy_kwargsp
- FEAT: ensure we can see baseline advantages in inspector.py
- BUG: PPO can solve FrozenLake-v1, but REINFORCE cannot. REINFORCE is likely not implemented correctly.
- TASK: solve Pong-v5_objects, then propagate to other envs
- BUG: run is being logged even when training is not started, bold terminal
- https://github.com/kenjyoung/MinAtar
- Add support for resuming training runs, this requires loading hyperparameters and schedulers to be timestep based; must also start from last timestep
- Change config files so that they only say their mention algo_id in experiment name
- TODO: compare atari breakout fps vs rlzoosb3
- WISHLIST: impelemnt SEEDRL with PPO to massively scale training
- can I create an exploration model by just making loss higher the more the model can predict the future?
- TODO: add config file beautifier that ensure attributes are set in the correct order
- TODO: add config file validator that ensures that all attributes are set and that they are set to the correct type
- runs: along with each checkpoint we are saving a json file with the metrics at that checkpoint. We also want to save a CSV with the rollout data for that epoch; this data should contain exactly the same data as the table in inspector.py, so we can encapsualte the function that generates the csv and reuse it in both places.
- inspector.py: add LLM debugging support
- BUG: checkpoint jsons not storing correct metrics
- add support for resuming training from a checkpoint
- FEAT: Add support for specifyingpo extra reward metrics for each environment, then make the reward shaper assign value for each of those rewards
- FEAT: add support for creating publishable video for youtube
- Generalized scheduling for metrics, make those be logged
- FEAT: train on cloud server (eg: lightning cloud)?
- Create benchmarking script to find optimal parallelization for env/machine combo
- TEST: predict next state to learn faster
- How to measure advantage of different seeds versus reusing same env.
- Consider increasing sample efficiency by figuring out how different are transitions between different envs
- FEAT: add determinism check at beginning to make sure that rollout benefits from multiple envs (eg: Pong, test on PongDeterministic)
- FEAT: create cartpole reward shaper that prioritizes centering the pole
- FEAT: track output distribution
- BUG: fix thread safety issues with async eval collector (copy model weights with lock)
- FEAT: consider computing mean reward by timesteps, this way in eval we just have to request n_steps = reward_threshold * N, this will make it easier to support vectorized envs
- FEAT: a2c (only after reinforce/ppo is stable)
- add support for plotting charts as text and feeding to llm, check how end of training does it
- https://cs.stanford.edu/people/karpathy/reinforcejs/index.html
- https://alazareva.github.io/rl_playground/
- FEAT: log results to huggingface
- FEAT: Write wandb diagnostics script, use claude desktop to debug
- FEAT: support continuous environments
- FEAT: support for multi-env rollout collectors
- FEAT: add multitask heads support (eg: Atari, Sega Genesis) -- consider large output space
- Ask agent for next learning steps/tasks (prompt file)
- REFACTOR: add type hints where applicable
- FEAT: add [Minari support](https://minari.farama.org/)
