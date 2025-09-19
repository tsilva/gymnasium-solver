Great question—there isn’t a single “right” number, but you can pick a good one quickly with a couple of measurements and a simple rule of thumb. Here’s a practical way to choose (works especially well for PPO/A2C-style on-policy training, then I’ll note off-policy differences).

How to pick n_envs (on-policy, e.g., PPO)
	1.	Decide your per-update batch size B.
Typical starting points: 2k–16k transitions per update (e.g., 4096 or 8192).
	2.	Pick a reasonable rollout length n_steps (per env).
Common: 128–256 (PPO) to balance GAE bias/variance and update frequency.
	3.	Compute an initial guess:
n_{\text{envs}} = \frac{B}{n_{\text{steps}}}
Example: B=8192, n_steps=256 → n_envs = 32.
	4.	Measure two times once (on your machine):
	•	t_env: average time for one env.step() (seconds/step) for your env.
	•	t_upd: time for one policy update on batch B.
	5.	Balance collection vs update time to avoid policy lag & idle GPU/CPU:
Collection time per update:
t_{\text{collect}} = \frac{B \cdot t_{\text{env}}}{n_{\text{envs}}}
Choose n_envs so that:
t_{\text{collect}} \lesssim t_{\text{upd}}
\quad\Rightarrow\quad
n_{\text{envs}} \approx \frac{B \cdot t_{\text{env}}}{t_{\text{upd}}}
Then round to something your hardware likes (e.g., multiple of cores) and ensure memory still fits.
	6.	Sanity constraints & quick heuristics
	•	Upper bound: don’t exceed what your CPU cores (or simulator threads) can actually run without thrashing; watch context switches.
	•	Lower bound: keep n_envs ≥ 8 for decent decorrelation (unless env is very heavy).
	•	Keep rollout length reasonable: if formula gives a huge n_envs, don’t shrink n_steps below ~64–128 just to make it fit; up the batch B instead.
	•	Throughput test: increase n_envs until total time per update stops dropping—then back off slightly.

Quick example
	•	Measured t_env = 0.6 ms/step (fast Atari-like), t_upd = 150 ms on B=8192.
	•	n_envs ≈ (8192 × 0.0006) / 0.150 ≈ 32.8 → try 32 or 48.
	•	With n_steps = 256, 32*256=8192 fits perfectly.

Off-policy (DQN/SAC/TD3, etc.)
	•	Parallel envs help decorrelate data and keep replay full, but too many can hurt sample efficiency due to policy lag between actors and learner.
	•	Start with 1–4 envs per actor, scale the number of actors instead of cranking a single vector env to 64+.
	•	Track replay ratio (grad steps / env steps). Keep it in a sensible range (often ~0.5–4 depending on algo). If replay ratio collapses when you add envs, you’ve gone too far.
	•	Keep the queue/buffer latency small (small batches more frequently from actors), and cap max staleness of experiences (e.g., prioritize recent samples or limit queue size).

Typical starting points by domain (rules of thumb)
	•	CPU physics (MuJoCo, Bullet): 16–32 envs for PPO with n_steps ≈ 128–256.
	•	Lite simulators (Atari, simple gridworlds): 16–64 envs.
	•	GPU-accelerated simulators (Isaac Gym, Brax, WarpDrive): hundreds to thousands (1k–16k) because stepping is massively parallel on GPU; tune B and n_steps accordingly.
	•	Very heavy envs (complex robotics, rendering): 4–16 envs; prioritize longer n_steps over more envs.

What to monitor while tuning
	•	Wall-clock improvement (time to reach score X), not just steps/sec.
	•	Policy lag indicators: performance degrades as n_envs grows, even though throughput improves.
	•	Update stability: exploding gradients or high variance? Consider increasing B or n_steps rather than n_envs.
	•	Utilization: aim for both simulator and learner (CPU/GPU) to be ~70–95% busy.

Minimal tuning loop (do this once per setup)
	1.	Choose B and n_steps → compute initial n_envs = B / n_steps.
	2.	Measure t_env and t_upd, set n_envs ≈ (B·t_env)/t_upd.
	3.	Try {½×, 1×, 2×} around that n_envs; pick the best reward vs time curve.
	4.	Lock it in; revisit if you change the model size, observation size, or hardware.

If you tell me your env, algo, and rough hardware, I can plug the numbers and give you a concrete n_envs to start with.