- add n_timesteps support
- add ability to hardcode reward threshold
- check that were are matching rlzoo stop criteria
solve mountaincar with framestacking
how do advnatges make it into final calc
rlzoo save model, run it in my model and compare rollout results (set seed and 1 env)
measure obs mean/var
make sure training uses same collecotrs so thry calc mean reward through their reward window, make sure it inits through config
expected steps to solve cartpole wirh reinforce, match that first
record videos in bg with model copy
check if rlzoo solves mountaincar
rlzoo better due to missing param support like decay
- save best model/agent checkpoints (use trainer) / background tasks records video 
- BUG: REINFORCE not working
- Try solving MountainCar-v0 with PPO + frame stacking (no reward shaping)
- Solve PongRAM-v0 with PPO
- BUG: RLZoo is doing 2x the FPS of this implementation, investigate
- Add max_grad_norm support (0.5 for cartpole)
- log stats like rlzoo
- mimick rlzoo hyperparam config structure
- Review CartPole-v1 hyperparams from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
- FEAT: log steps/episodes to progress bar
- FEAT: log experiment data at start of training
- REFACTOR: move reward threshold detection to rollout collector
- FEAT: Log config into wandb experiment
- FEAT: eval model on every n steps
- BUG: REINFORCE is calculating returns using value model?
- BUG: eval is being calculated before window is full, consider evaling frequently by n_steps instead of n_episodes
- BUG: fix thread safety issues with async eval collector (copy model weights with lock)
- FEAT: get reward threshold from env specs (hardcoded)
- FEAT: use torch.inference_mode() where applicable
- FEAT: support for softcoding activations
- FEAT: train for convergence without deterministic policy
- FEAT: add baseline subtraction to A2C
- FEAT: a2c (only after reinforce/ppo is stable)
- FEAT: add normalization support
- FEAT: track output distribution
- FEAT: support continuous environments
- FEAT: support for multi-env rollout collectors
- FEAT: add multitask heads support (eg: Atari, Sega Genesis) -- consider large output space
- FEAT: add support for N PPO updates per rollout
- CHECK: run rollout through dataloader process, do we always get n_batches?
- CHECK: assert that dataloader is always going through all expected batches
- TODO: solve Pong-RAM with PPO
- Add value baseline support for reinforce
- Match sb3rlzoo metric names
- Match sb3rlzoo performance
- When I group by episodes I discard data from that rollout that won't be included in the next sequence of trajectories, so I need to make sure I don't lose data
- Log "n_steps" and "n_episodes" in the metric tracker
- FEAT: add alert support to metric tracker (algo dependent)
- make best model be saved to wandb
- log results to huggingface?
- TODO: make evaluation run in background and keep a mean reward window, it picks up the model params set up in it, runs N envs in sequence with N workers and 
- Write wandb diagnostics script, use claude desktop to debug
- add environment normalization support
- benchmark against rlzoo with same hyperparameters
- add support for plotting charts as text and feeding to llm, check how end of training does it
- track environment stats, observarion stats, reward distributions, etc
- change api to match sb3
- https://cs.stanford.edu/people/karpathy/reinforcejs/index.html
- https://alazareva.github.io/rl_playground/