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
- Don't bother with running evals until you reach treshold on training.
- ✅ Max out n_envs as far as CPU allows — cheap way to scale diversity and speed.
- ✅ Set n_steps long enough to capture temporal structure, but short enough to avoid stalling updates.
- ✅ For REINFORCE, aim for larger effective rollout sizes (n_envs × n_steps) to tame high-variance returns.
- ✅ Use reward-to-go and baselines to cut variance further instead of only relying on bigger rollouts.




--- 

Great question — batch size is the third lever alongside n_envs and n_steps, since it controls how much of your collected data is actually used in one gradient update.

Here are the rules of thumb for batch sizes in policy gradient / actor–critic settings:

⸻

📌 Batch Size (data used per gradient update)
	•	✅ Larger batch sizes
	•	Lower gradient variance → more stable learning.
	•	More efficient use of GPU (parallel matrix ops).
	•	Especially useful for on-policy methods (PPO, REINFORCE) where sample efficiency is limited.
	•	⛔ Too large batch sizes
	•	Slower iteration speed (fewer updates per unit of data).
	•	Risk of underfitting if learning rate is not scaled accordingly.
	•	Memory bottlenecks.
	•	✅ Smaller batch sizes
	•	More frequent updates → faster reaction to data.
	•	Potentially better exploration through noisier gradients.
	•	⛔ Too small batch sizes
	•	Extremely noisy gradients → unstable training.
	•	Wastes parallel compute since GPUs run less efficiently on tiny batches.

⸻

📌 Rules of thumb
	•	Batch size ≈ 32–256 per update is a good starting range for most RL workloads (mirrors supervised learning practice).
	•	In PPO / A2C, set batch_size as a fraction of n_envs × n_steps (commonly minibatches = 4–32 splits).
	•	A good heuristic:
	•	Keep batch_size ≥ action_dim × 10 to avoid overly noisy gradients.
	•	Ensure each batch contains samples from multiple environments (so you’re not overfitting to one trajectory slice).
	•	For REINFORCE / high-variance returns, bigger batches (≥1k samples) are often necessary for stable updates.
	•	For continuous control (MuJoCo, PyBullet, etc.), ~2–10k samples per update (via multiple envs × steps, split into minibatches) tends to be standard.

⸻

⚖️ Balance tip:
	•	Effective rollout size = n_envs × n_steps.
	•	From this pool, choose batch_size so that you have at least 4–10 minibatches per epoch.
	•	Example: if rollout size = 8192, set batch_size = 256 → 32 minibatches per epoch.

⸻

👉 Do you want me to make a summary diagram showing how n_envs, n_steps, and batch_size interact (like a flow of rollout → minibatching → updates)? That could complement your guide nicely.


0) Set the ground rules (once)

Determinism & logging

Fix seeds; turn off nondeterministic ops; log everything (returns, entropy, grad-norm, loss, FPS).

Keep an evaluation protocol: every N updates, run 5–10 eval episodes with exploration off; track mean/median and IQR.

Throughput check

Profile sample collection vs. update time. Aim for ≥70% GPU/TPU utilization during updates and no long GPU idle gaps.

1) Max out data pipeline

n_envs (parallel envs)

Increase until CPU is ~80–90% loaded or envs start thrashing RAM.

If GPU idles while collecting, add envs before touching anything else.

Rollout sizing (batch_size for REINFORCE)

REINFORCE is high-variance → start with bigger updates.

Starting points

Discrete control: 2–8k timesteps/update.

Continuous control: 5–20k timesteps/update.

Ensure ≥16–64 episodes per batch (use many envs so you’re not dominated by a few long trajectories).

If returns swing wildly across updates: double batch; if learning crawls: halve batch.

Tip: If you’re also using n_steps, choose it to keep updates frequent (e.g., collect until batch_size is met across envs) rather than requiring full episodes.

2) Core optimizer sweep

Learning rate (policy)

Log-sweep: {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} (Adam).

Pick by area under the learning curve (sample-efficiency + final return).

If you increase batch size, either keep LR and resweep, or try ~linear-ish scaling up to ~4× before resweeping.

Epochs per batch

REINFORCE can overfit a batch quickly. Try {1, 2, 4}; prefer 1–2 if you see training loss collapse while eval return stalls.

Gradient clipping

Global-norm {0.5, 1.0, 2.0}.

If you see sporadic loss spikes or NaNs: lower clip or LR.

3) Variance reduction (must-have)

Reward-to-Go

Use it. It strictly reduces variance vs. full-episode returns. No sweep needed.

Baseline (state-value) & advantage norm

Add a learned V(s) baseline (a tiny MLP).

Train with MSE; value-loss coeff ~0.5; value LR = policy LR (try {1×, 2×}).

Normalize advantages per batch (zero mean, unit std). Always on—don’t sweep.

Reward/return normalization

If reward scales drift across tasks or time, normalize returns by a running std (or PopArt). Turn on if learning is twitchy.

4) Horizon & exploration

Discount factor γ

Start 0.99.

Use the horizon heuristic: target effective H ≈ 1/(1−γ). For task horizon T, try γ ≈ 1 − 1/(0.25T … 1.0T).

Sweep small set: {0.95, 0.98, 0.99, 0.995, 0.999}.

Symptoms: too myopic → improve with higher γ; credit assignment too fuzzy/slow → lower γ.

Entropy bonus (exploration)

Start 0.01 (discrete) / 0.001 (continuous). Sweep {×0.3, ×1, ×3}.

Then anneal to 0 over the last 50–80% of training.

If policy collapses early: raise entropy or slow the anneal.

5) Finishing touches

Optimizer betas / weight decay

Adam β2 0.999 (default). If value loss lags, try β2 0.99.

Tiny L2 (e.g., 1e-4) only if you see overfitting on short horizons.

Observation normalization

Running mean/var per state dim. Essential for continuous control or nonstationary rewards.

Learning-rate schedule

Cosine or linear decay to 0; warmup 0–2k updates if gradients are spiky at start.

6) How to run the tuning (playbook)

Phase A (throughput): tune n_envs → pick a stable batch_size (Section 1).

Phase B (stability): sweep LR × epochs × clip (Sections 5–7).

Phase C (variance): enable baseline + adv-norm; if still noisy, add return normalization (Sections 9–10).

Phase D (horizon/exploration): sweep γ × entropy small grids (Sections 11–12).

Phase E (polish): LR schedule, observation norm, tiny L2 (Section 13–15).

Budgeting: use coarse sweeps first (1–2 seeds, 0.5–1h runs), then re-run top 2–3 configs with 3–5 seeds for your final pick.

7) Quick starting configs

Discrete (e.g., Atari-like):

batch_size 4–8k, epochs 1–2, LR 3e-4, γ 0.99, entropy 0.01→0, clip 1.0.

Continuous (e.g., MuJoCo):

batch_size 10–20k, epochs 1–2, LR 1e-4–3e-4, γ 0.99–0.995, entropy 0.001→0, clip 1.0.

8) Debug/triage cheatsheet

Divergence/NaNs → lower LR, add/raise clip, bigger batch, check obs/reward norms.

Wild update-to-update swings → bigger batch, advantage/return norm, stronger baseline.

Stuck learning → raise LR a notch, add epochs (to 2), increase entropy (temporarily), or bump γ.

Overfitting a batch (train loss ↓, eval flat) → fewer epochs (1), larger batch, stronger entropy.

If you want, I can turn this into a one-page checklist you can drop into your repo’s README.