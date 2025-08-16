- FEAT: gradio app that allows running a given run id and see the frames, actions, rewards, etc. This will allow us to debug runs and see how the agent behaves in different scenarios. This feature should be implemented in a file called inspect.py. You can call it by providing the run id as an argument, defaulting to the latest run. The app should allow selecting the checkpoint step to inspect and should display the frames, actions, rewards, etc. in a user-friendly way. By default use the best checkpoint, if not available use the latest.
- FEAT: Whenever we record a checkpoint we need to store all metrics at that point.
- FEAT: call challenges instead of environments
- FEAT: Add support for specifyingpo extra reward metrics for each environment, then make the reward shaper assign value for each of those rewards

- FEAT: add support for creating publishable video for youtube

- Generalized scheduling for metrics, make those be logged
- FEAT: track learning hyperparams in wandb (train/hyperparams)

- FEAT: When training ends show ascii plot of all metrics
- FEAT: Add support for generating end of training report
- BUG: action distributions are note being logged correctly (how will I log this along with our current per epoch system?)
- FEAT: ask copilot to create its own isntructrions namely to generate its own techical documentation that it keeps up to date
- REFACTOR: rollout buffer can be much more efficient (review how sb3 does it) -- our fps is still 1000 below sb3
- FEAT: add ability for the cursor agent to be able to run and adjust hyperparams by itself
- TODO: Figure out why CartPole-v1/PPO works better with Tahn activation than ReLU
- BUG: PPO can solve FrozenLake-v1, but REINFORCE cannot. REINFORCE is likely not implemented correctly.
- BUG: is REINFORCE well implemented? are we waiting until the end of the episode to update the policy?
- FEAT: track immediate episode reward (for monitoring hyperparam change reaction)
- FEAT: improve metric descriptions
- FEAT: write guide on how to monitor training
- FEAT: train on cloud server
- FEAT: normalize rewards?
- FEAT: add normalization support
- FEAT: add discrete env support
- Create benchmarking script to find optimal parallelization for env/machine combo
- TEST: predict next state to learn faster
- FEAT: add [Minari support](https://minari.farama.org/)
- BUG: videos not logged at correct timesteps
- FEAT: improve config structurtee
- FEAT : Normalize returns for REINFORCE
- FEAT: add warning confirming if ale-py has been compiled to target architecture (avoid rosetta in silicon macs)
- BUG: baseline_mean/std are being recorded even when not used
- BUG: metrics are not well sorted yet
- FEAT: add support for fully recording last eval
- BUG: check that were are matching rlzoo mstop criteria
- Pong-RAM: Add support for byte-selection
MaxAndSkipEnv
- BUG: video step is still not alligned
- BUG: step 100 = reward 99
- FEAT: add vizdoom support
- FEAT: add stable-retro support
- Adjust eval policy for atari
	•	Normalize each byte (e.g., to [0,1]) and consider embedding bits (treat bytes as 8 bits).
	•	Try longer training and more seeds; RAM setups often need more steps to stabilize.
	•	If you can, expose extra emulator registers (RAM+) to reduce partial observability.
- How to measure advantage of different seeds versus reusing same env.
- Consider increasing sample efficiency by figuring out how different are transitions between different envs
- FEAT: add determinism check at beginning to make sure that rollout benefits from multiple envs (eg: Pong, test on PongDeterministic)
- FEAT: create cartpole reward shaper that prioritizes centering the pole
- FEAT: track output distribution
- BUG: fix thread safety issues with async eval collector (copy model weights with lock)
- FEAT: consider computing mean reward by timesteps, this way in eval we just have to request n_steps = reward_threshold * N, this will make it easier to support vectorized envs
- FEAT: a2c (only after reinforce/ppo is stable)
- FEAT: support for softcoding activations
- FEAT: add same linear decay features as rlzoo
- FEAT: add interactive mode support
- CHECK: run rollout through dataloader process, do we always get n_batches? assert it 
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

- REINFORCE: