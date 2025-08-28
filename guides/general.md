📌 n_envs (number of parallel environments)
- ✅ Increases rollout size.
- ✅ More diverse data → less correlated samples → cleaner gradient estimates.
- ✅ Faster data collection via parallelization.
- ⛔ Higher memory usage.
- ⛔ Improves only spatial diversity (different states at same timestep), not temporal credit assignment.

⸻

📌 n_steps (steps per environment before update)
- ✅ Increases rollout size.
- ✅ Better temporal credit assignment (exposes policy to longer returns).
- ✅ Reduces bootstrap bias in actor–critic methods.
- ⛔ Consecutive samples are correlated (less diverse).
- ⛔ Slows updates (must wait for all steps before training).
- ⛔ Risk of GPU idle time if rollout collection is CPU-bound.

⸻

📌 Rules of thumb
- To ensure reproducibility, perform two runs in a row and ensure all wandb graphs are identical.
- ✅ Max out n_envs as far as CPU allows — cheap way to scale diversity and speed.
- ✅ Set n_steps long enough to capture temporal structure, but short enough to avoid stalling updates.
- ✅ For REINFORCE, aim for larger effective rollout sizes (n_envs × n_steps) to tame high-variance returns.
- ✅ Use reward-to-go and baselines to cut variance further instead of only relying on bigger rollouts.
