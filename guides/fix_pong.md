Quick diagnosis
- Policy saturation: entropy fell from ~0.35 early to ~0.26–0.28; opt/ppo/clip_fraction ~0.05; opt/ppo/approx_kl ~0.002–0.004. PPO updates are very conservative; advantages partly washed out.
- Value head underfits near plateau: opt/value/explained_var drops to ~0.68 around the stall; opt/loss/value rising slightly.
- Rollout horizon might be short for sparse scoring: n_steps 256 across 16 envs (4096) works, but episodes last ~4k–4.4k steps on objects features; bootstrapping dominates, making advantage signal small.
- Objects feature vector normalization may clip informative extremes: wrapper clips to [0,1]; MIN/MAX are from random play; after competence increases, ball/paddle dynamics can exceed calibration.

Concrete fixes (apply 2–4 at once, then re-evaluate after ~1–2M steps)
- Exploration and update size
  - ent_coef: increase slightly to 0.005–0.008. Start 0.006.
  - policy_lr: add linear decay: policy_lr: lin_3e-4 to keep early speed but reduce late staleness.
  - clip_range: widen to 0.2 with linear decay (clip_range: 0.2, clip_range_schedule: linear). This raises clip_fraction toward ~0.1–0.2.
- Advantage signal and batch quality
  - n_steps: 512 (rollout size 8192) to improve advantage SNR on long rallies.
  - n_epochs: 4–6 instead of 10 to reduce overfitting per batch as rollout size grows. Try 6.
  - batch_size: keep 1024 (8 minibatches of 8192). If GPU room, 2048.
  - normalize_advantages: keep "batch".
- Value head
  - vf_coef: 0.7 to prioritize critic fit near plateau.
  - hidden_dims: [256, 256] for more capacity on non-linear object dynamics.
- Observation features
  - Disable clipping in `PongV5_FeatureExtractor` (clip: false) so normalized features can exceed [0,1] if calibration MIN/MAX are off.
  - Optionally recompute MIN/MAX from a longer random policy sweep, or enable `normalize_obs: true` in config to stabilize feature scale.
- Evaluation cadence
  - eval_freq_epochs: 100 to reduce eval overhead (not a big factor, but keeps focus on training).

Minimal edit to config (ALE-Pong-v5_objects_ppo)
- n_steps: 512
- n_epochs: 6
- clip_range: 0.2
- clip_range_schedule: linear
- ent_coef: 0.006
- policy_lr: lin_3e-4
- vf_coef: 0.7
- hidden_dims: [256, 256]
- Optionally add to env_wrappers list: { id: PongV5_FeatureExtractor, clip: false } to override default clipping.

Next steps I recommend
- Spawn a new run with the above changes; monitor:
  - train/opt/ppo/clip_fraction target 0.1–0.2
  - train/opt/ppo/approx_kl ~0.01–0.02 max
  - train/opt/value/explained_var trending back >0.8
  - train/roll/ep_rew/mean slope over next 1–2M steps
- If still flat, try n_envs 32 with n_steps 256 (same batch size 8192) to diversify episodes.
