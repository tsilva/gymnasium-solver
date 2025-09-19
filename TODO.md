# TODO

## NEXT

- BUG: wandb run is being logged even when training is not started, bold terminal
- TODO: optimal single metric for best training (samples/reward)
- TODO: dont tie eval_freq_epochs and eval_warmup_epochs to epochs because ppos updates for many epochs; consider changing eval_freq to timesteps so it doesnt depend on n_updates
- FEAT: prefix metrics (eg: time/epoch, time/step, etc., env/obs_mean, env/obs_std, env/)
- FEAT: make boot and shutdown times faster
- FEAT: wandb bug may caused by lack of .finish()
- FEAT: add kl early stopping to PPO
- BUG: can't inspect bandit training
- BUG: W&B workspace redundant print
- FEAT: add normalization support 
- TEST: are sweeps still working?
- BUG: reset optimizer on learning rate changes
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
- FEAT: support continuous environments
- FEAT: support for multi-env rollout collectors
- FEAT: add curriculum learning support
- FEAT: add multitask heads support (eg: Atari, Sega Genesis) -- consider large output space
- REFACTOR: consolidate single-trajectory return/advantage helpers by moving run_inspect.py:compute_mc_returns and run_inspect.py:compute_gae_advantages_and_returns onto shared utilities next to utils/rollouts.py:compute_batched_mc_returns/compute_batched_gae_advantages_and_returns; acceptance: inspector outputs stay the same and rollout-related tests remain green.
- REFACTOR: deduplicate YAML config discovery by reusing Config loader logic between scripts/smoke_test_envs.py:_collect_env_config_map/_resolve_inheritance and utils/config.py:Config.build_from_yaml via a shared utility (e.g., utils/config_loader.collect_raw_configs); acceptance: both callers produce identical config maps without manual field lists and existing config-selection tests still pass.

## WISHLIST
 
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