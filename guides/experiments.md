Got it—pure policy-gradient progression only: VPG (a.k.a. REINFORCE) → A2C → PPO. Below is a fail → fix syllabus you can replicate. Each env starts with a VPG failure you’ll see, then the fixes with A2C and finally PPO. I’ve updated IDs/targets (e.g., LunarLander-v3) and included concrete configs + budgets + expected outcomes.

Conventions
seed=42, RecordEpisodeStatistics, eval over 50–100 episodes (deterministic). Targets from Gymnasium docs: CartPole-v1 threshold 500; LunarLander solved at 200 (now v3); Acrobot threshold −100; BipedalWalker target 300; Pendulum reward ∈ [−16.27, 0] (closer to 0 is better).  ￼ ￼

⸻

1) CartPole-v1 (discrete, short horizon)

Fail (VPG-naive)
	•	Algo: REINFORCE without baseline, without reward-to-go.
	•	Net: [64,64], ReLU. lr=1e-3, gamma=0.99, batch ≈10k steps, no entropy.
	•	Budget: 150–250k steps.
	•	What you’ll see: jagged learning, frequent collapses, often plateaus < 400.

Fix 1 (VPG + tricks)
	•	Keep REINFORCE but add: reward-to-go, value baseline (MC advantage), advantage normalization, entropy=0.001–0.005.
	•	Same budget: 150–250k.
	•	Target: mean return ≈ 475–500 (v1 threshold 500).  ￼

Fix 2 (A2C)
	•	SB3-style: n_steps=5, gamma=0.99, gae_lambda=1.0 (vanilla A2C), lr=7e-4, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs={"net_arch":[64,64]}.
	•	Budget: 150–250k.
	•	Expected: reaches ~500 more reliably than VPG.

Fix 3 (PPO)
	•	n_steps=1024, batch_size=64, n_epochs=10, lr=3e-4, clip=0.2, gae_lambda=0.95, ent_coef=0.0, [64,64].
	•	Budget: 100–200k.
	•	Expected: stable ~500 with smooth curve.  ￼

⸻

2) Acrobot-v1 (discrete, long horizon, sparse-ish)

Fail (VPG + tricks still struggles)
	•	REINFORCE with baseline + reward-to-go + entropy 0.01, lr=1e-3, [128,128].
	•	Budget: 1–2M.
	•	What you’ll see: oscillates between −200…−120, seldom crosses −100.

Fix 1 (A2C)
	•	n_steps=5, gamma=0.99, gae_lambda=1.0, lr=7e-4, vf_coef=0.5, ent_coef=0.0, [128,128].
	•	Budget: 500k–1M.
	•	Target: ≥ −100 mean over 100 eps (reward threshold).  ￼

Fix 2 (PPO)
	•	n_steps=2048, batch_size=64, n_epochs=10, lr=3e-4, clip=0.2, gae_lambda=0.95, ent_coef=0.01, [128,128].
	•	Budget: ~500k–1M.
	•	Expected: steadier path to ≥ −100.  ￼

⸻

3) LunarLander-v3 (discrete Box2D, medium horizon)

Fail (VPG even with tricks)
	•	REINFORCE + baseline + reward-to-go + entropy 0.01, lr=3e-4, [128,128].
	•	Budget: 2–4M.
	•	What you’ll see: climbs to 150–200, crashes/regresses due to high variance & large policy steps.

Fix 1 (A2C)
	•	n_steps=5, gamma=0.99, gae_lambda=1.0, lr=7e-4, vf_coef=0.5, ent_coef=0.01, [128,128].
	•	Budget: 1–2M.
	•	Expected: better stability; may flirt with 200 but still bouncy.

Fix 2 (PPO)
	•	n_steps=2048, batch_size=64, n_epochs=10, lr=3e-4, clip=0.2, gae_lambda=0.95, ent_coef=0.01, [128,128].
	•	Budget: ~1.0M.
	•	Target: ≥ 200 mean over 100 eps on LunarLander-v3 (use v3; see version notes).  ￼

⸻

4) Pendulum-v1 (continuous, dense but tricky exploration)

Fail (VPG Gaussian)
	•	Gaussian policy, MC advantage baseline, lr=3e-4, entropy 0.0, [256,256].
	•	Budget: 500k–1M.
	•	What you’ll see: slow learning, returns often stuck below −300.

Fix 1 (A2C, continuous)
	•	n_steps=5, gamma=0.99, gae_lambda=1.0, lr=3e-4, vf_coef=0.5, ent_coef=0.0, [256,256].
	•	Budget: 300k–600k.
	•	Expected: average return improves to ≥ −200.

Fix 2 (PPO, continuous)
	•	n_steps=2048, batch_size=64, n_epochs=10, lr=3e-4, clip=0.2, gae_lambda=0.95, ent_coef=0.0, [256,256].
	•	Budget: ~300k.
	•	Target: ≥ −200 average; remember optimal is 0, min per-step ≈ −16.27.  ￼

⸻

5) BipedalWalker-v3 (continuous, contact-rich, long horizon)

Fail (VPG Gaussian + tricks)
	•	REINFORCE Gaussian + baseline + entropy 0.01, lr=3e-4, [256,256].
	•	Budget: 3–5M.
	•	What you’ll see: flat (< 50) or sporadic spikes that don’t persist—variance explosion.

Fix 1 (A2C)
	•	n_steps=5, gamma=0.99, gae_lambda=1.0, lr=3e-4, vf_coef=0.5, ent_coef=0.0, [256,256].
	•	Budget: 3–5M.
	•	Expected: progress but unstable; may not consistently solve.

Fix 2 (PPO)
	•	n_steps=4096, batch_size=128, n_epochs=10, lr=3e-4, clip=0.2, gae_lambda=0.95, ent_coef=0.0, [256,256].
	•	Extras that help here: observation normalization, reward scaling, gradient clipping (0.5–1.0).
	•	Budget: 3–5M.
	•	Target: ≥ 300 on normal walker (solve criterion).  ￼

⸻

Why each “fix” helps (policy-gradients only)
	•	VPG → VPG+tricks: reward-to-go + learned baseline + advantage norm cut variance; entropy resists premature determinism.
	•	VPG → A2C: Online advantage from a critic reduces variance further and improves credit assignment over long horizons (no replay, still PG).
	•	A2C → PPO: clipped surrogate + multiple epochs + GAE control update size, yielding smoother, more reliable improvement (trust-region flavor).

⸻

Ready-to-run SB3 configs (A2C/PPO)

A2C (discrete or continuous)

from stable_baselines3 import A2C
model = A2C("MlpPolicy", env,
            n_steps=5, gamma=0.99, gae_lambda=1.0,
            learning_rate=7e-4, vf_coef=0.5, ent_coef=0.0,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[64,64]), seed=42, verbose=0)
model.learn(total_timesteps=500_000)  # set per env above

PPO (discrete or continuous)

from stable_baselines3 import PPO
model = PPO("MlpPolicy", env,
            n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=[128,128]),
            seed=42, verbose=0)
model.learn(total_timesteps=1_000_000)  # set per env above

Env references for targets/versions: CartPole-v1 (threshold 500), Acrobot-v1 (threshold −100), LunarLander-v3 version notes (use v3), BipedalWalker-v3 solve 300, Pendulum reward range [−16.27, 0].  ￼ ￼

⸻

Tips so your fail → fix is obvious
	•	Log policy entropy, advantage variance, and grad norms alongside returns.
	•	Keep identical nets & seeds across VPG/A2C/PPO within each env.
	•	Prefer vectorized envs (4–8) for A2C/PPO; keep VPG single-env to feel the variance pain.
	•	Use early-stop eval: if you hit the target, freeze training and evaluate 100 eps.

If you want, I can paste a tiny VPG script you can toggle (baseline on/off, reward-to-go on/off, entropy on/off) to reproduce the failures exactly, matching the SB3 A2C/PPO settings for the “fix” runs.