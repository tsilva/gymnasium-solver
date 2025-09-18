Cleanup Targets

  - inspector.py:294 & inspector.py:696 – run_episode and build_ui are 1000+ lines together, bundling env/model setup, data munging, and Gradio wiring into huge nested
  closures; they’re hard to reason about compared with BaseAgent’s structure.
  - utils/rollouts.py:434 (see also :497, :828, :940) – the collector is a 1k-line grab bag of rollout storage, statistics, evaluation, and logging with many TODO
  “review this” notes; logic like collect() still ships obvious hacks.
  - utils/logging.py:70 – the helper declares __getattr__ indented under strip_ansi_codes, so it never runs, and the module mixes tee streams, log rotation, and ANSI
  utilities without clear separation.
  - trainer_callbacks/end_of_training_report.py:100 – on_fit_end remains a single 150+ line method (flagged by its own TODO), doing ad-hoc introspection, CSV parsing,
  templating, and filesystem writes inline.
  - utils/models.py:30 & utils/models.py:174 – the MLP builders rely on type lists, numpy dtypes, and squeeze hacks with TODOs complaining about them; actor-critic
  shape handling needs a principled cleanup.

  Next steps: 1) break the inspector helpers into smaller modules/functions (env/model loading, UI layout, event handlers). 2) carve RolloutCollector into focused
  components (buffer, metrics, evaluation) before touching the TODO-labeled logic.

- must sort by alert id not by metric
- FEAT: count how many times each alert triggered, couple that with predefined recommendations on what to tune
- FEAT: prefix metrics (eg: time/epoch, time/step, etc., env/obs_mean, env/obs_std, env/)
- FEAT: make boot and shutdown times faster
- FEAT: wandb bug may caused by lack of .finish()
- FEAT: wandb run names (as they appear in dashboard) are currently {algo_id}-{seed}, make them {algo_id}-{run_id}
- BUG: can't inspect bandit training
- BUG: W&B workspace redundant print
- FEAT: make time elapsed metric be highlited as blue
- FEAT: log model gradients to wandb
- FEAT: add autotuner worklflow (check wandb logs)

- FEAT: use codex-cli to debug runs
- Ensure metrics summary is sorted with metrics.yaml priorities
- FEAT: add normalization support 
- FEAT: add sweeping support
- TODO: how is pongv5 determinism enforce
- TODO: run pongv5 sweep
- REFACTOR: simplify CNN policy creation, test manually first
- BUG: restore grayscaling / resizing logic; think how to unify with atari preprocessing (probably just inspect internals and extract)
- FEAT: add kl early stopping to PPO
- BUG: fix cnn policy training, in separate file create env and inspect traversing layers
- FEAT: overridable rollout collector factory methods instead of extra parameter method
- FEAT: log checkpoint times to wandb
- BUG: reset optimizer on learning rate changes
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
- FEAT: add curriculum learning support
- FEAT: add multitask heads support (eg: Atari, Sega Genesis) -- consider large output space
- Ask agent for next learning steps/tasks (prompt file)
- REFACTOR: add type hints where applicable
- FEAT: add [Minari support](https://minari.farama.org/)
