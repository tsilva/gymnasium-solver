- Adjust eval policy for atari
	•	Stack k RAM frames (e.g., 4–8) and add a small LSTM/GRU on top of the MLP.
	•	Normalize each byte (e.g., to [0,1]) and consider embedding bits (treat bytes as 8 bits).
	•	Increase model capacity slightly (wider MLP) and retune PPO (lr, clip range, entropy bonus, batch size).
	•	Try longer training and more seeds; RAM setups often need more steps to stabilize.
	•	If you can, expose extra emulator registers (RAM+) to reduce partial observability.
- How to measure advantage of different seeds versus reusing same env.
- Add OCAtari support
- Try creating local run folder with assets, create own run id and assign it to wandb if possible
- Consider increasing sample efficiency by figuring out how different are transitions between different envs
- FEAT: add determinism check at beginning to make sure that rollout benefits from multiple envs (eg: Pong, test on PongDeterministic)
- FEAT: add logging support (file logging)
- FEAT: create cartpole reward shaper that prioritizes centering the pole
- FEAT: add assertions where applicable
- FEAT: add support for fully recording last eval
- FEAT: track output distribution
- BUG: videos not logged at correct timesteps
- BUG: confirm that buffer growth is under control
- BUG: fix thread safety issues with async eval collector (copy model weights with lock)
- BUG: check that were are matching rlzoo stop criteria
- FEAT: consider computing mean reward by timesteps, this way in eval we just have to request n_steps = reward_threshold * N, this will make it easier to support vectorized envs
- FEAT: support for softcoding activations
- FEAT: a2c (only after reinforce/ppo is stable)
- FEAT: add same linear decay features as rlzoo
- FEAT: add support for premature early stop if train_mean_reward is above threshold
- FEAT: add interactive mode support
- rlzoo save model, run it in my model and compare rollout results (set seed and 1 env)
- rlzoo better due to missing param support like decay
- CHECK: run rollout through dataloader process, do we always get n_batches? assert it 
- add support for plotting charts as text and feeding to llm, check how end of training does it
- change api to match sb3
- https://cs.stanford.edu/people/karpathy/reinforcejs/index.html
- https://alazareva.github.io/rl_playground/
- FEAT: log results to huggingface
- FEAT: Write wandb diagnostics script, use claude desktop to debug
- FEAT: add normalization support
- FEAT: support continuous environments
- FEAT: support for multi-env rollout collectors
- FEAT: add multitask heads support (eg: Atari, Sega Genesis) -- consider large output space
- Ask agent for next learning steps/tasks (prompt file)
- REFACTOR: add type hints where applicable