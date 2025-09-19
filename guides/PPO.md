# PPO Setup & Tuning Checklist

## 1. Fix constraints & goals
- **Budget:** how many env steps can you afford? (e.g., 5–20M)  
- **Throughput:** how many parallel envs can you run? (CPU-bound)  
- **Memory:** rough cap for a single minibatch on your accelerator/CPU.  
- **Horizon/rewards:** short vs long horizon, dense vs sparse reward.  

---

## 2. Pick a sane starting point (by domain)
- **Short-horizon / fast envs** (Atari, gridworld, simple sim):  
  - `num_envs = 8–32`, `n_steps = 128–256` → rollout = 2k–8k  
  - `minibatch_size = 256–1024`, `epochs = 3–4`  

- **Continuous control** (MuJoCo, robotics sim):  
  - `num_envs = 1–8`, `n_steps = 1024–4096` → rollout = 4k–32k  
  - `minibatch_size = 1024–4096`, `epochs = 5–10`  

- **Procedurally generated / high-var envs** (Procgen-like):  
  - `num_envs = 32–128`, `n_steps = 128–256` → rollout = 8k–32k  
  - `minibatch_size = 1024–4096`, `epochs = 1–3`  

**Rule of thumb:** aim for total rollout per update = **10k–50k** samples early on.  

---

## 3. Compute the core quantities
- **Total rollout per update:** `N_total = n_steps × num_envs`  
- **#Minibatches per epoch:** `M = N_total / minibatch_size` (make it integer)  
- **Replay/UTD ratio (on-policy PPO):** ≈ `epochs` (each sample reused `epochs` times)  
- **SGD steps per update:** `epochs × M`  

---

## 4. Set the stabilizers (good defaults)
- **Advantage:** GAE `λ = 0.95`, `γ = 0.99`  
- **Optimizer:** AdamW, `lr = 3e-4` (decay over time)  
- **Clip:** `ε = 0.1–0.2` (decay a bit over time)  
- **Entropy bonus:** `0.0–0.01` (sparser rewards → lean higher)  
- Normalize observations & advantages; consider reward normalization for high-var tasks.  

---

## 5. Make it fit & fast
- If **OOM** → decrease `minibatch_size` first (keep `N_total` unchanged).  
- If **GPU/CPU idle** → increase `minibatch_size` (or `N_total`).  
- Keep `n_steps ≥ 128` (too small → noisy advantages).  
- Keep `n_steps ≤ 4096` (too big → slow updates).  

---

## 6. Bring-up checklist (“does it learn at all?”)
- Start with domain defaults.  
- Verify logs update cleanly (no NaNs).  
- Returns trend up on at least 1–2 seeds.  
- If unstable: halve `lr` or increase `minibatch_size`; optionally lower `epochs`.  

---

## 7. Tune rollout size first
- **Increase `N_total` when:** returns bounce wildly, advantages noisy, sparse/delayed rewards.  
- **Decrease `N_total` when:** learning is stable but wall-clock is slow.  
- **Typical sweet spot:** 16k–32k samples/update.  

---

## 8. Tune minibatch size second
- Keep `minibatch_size ≈ 1/4 – 1/16` of `N_total`.  
- Larger minibatches → stabler grads, better throughput.  
- Smaller minibatches → faster per-step, more noise.  
- Ensure divisibility: `N_total % minibatch_size == 0`.  

---

## 9. Tune epochs last
- Start with `epochs = 3–10`.  
- **Underfitting (low clip frac, tiny KL, slow improvement):** ↑epochs.  
- **Over-updating (KL spikes, clip frac > 0.5, collapse):** ↓epochs.  

---

## 10. What to watch in logs
- **Episode return (eval):** should trend up. Plateaus → ↑N_total or ↓lr.  
- **Explained variance (value fn):** aim for >0.6 mid-training. Too low → ↑N_total, ↓lr, or ↑value net.  
- **Approx KL:** aim for ~0.01–0.05 per update.  
  - Too high → ↓lr, ↓epochs, ↑clip ε.  
  - Too low → ↑lr, ↑epochs, ↓clip ε.  
- **Clip fraction:** healthy 10–30%.  
  - ~0% → updates too timid (↑lr/epochs, ↓ε).  
  - 50% → too aggressive (↓lr, ↑ε, ↓epochs).  
- **Entropy:** should decay gradually. If crashes to 0 early → ↑entropy bonus.  

---

## 11. Simple schedules
- **Learning rate decay:** linear or cosine to ~10–20% of start.  
- **Clip range decay:** e.g., 0.2 → 0.1 across training.  
- (Keep batch/minibatch fixed; adjust `N_total` only if needed.)  

---

## 12. Small, reliable tuning grid
Run 2×2×2 grid with 3 seeds each:  
- `N_total ∈ {16k, 32k}`  
- `minibatch_size ∈ {1024, 2048}` (divides `N_total`)  
- `epochs ∈ {4, 8}`  
→ Pick best mean eval return @ fixed samples.  

---

## 13. Quick “symptoms → fixes”
- **Unstable / oscillating returns:** ↑N_total, ↓lr, ↓epochs, ↑ε.  
- **No learning / flat returns:** ↑lr, ↑epochs, ↑entropy bonus, or ↑N_total for sparse rewards.  
- **Value loss explodes / bad EV:** ↓lr, ↑N_total, check normalization.  
- **KL spikes / early collapse:** set `target_kl = 0.02–0.05` in the PPO config to stop extra minibatch epochs once KL drifts too far, ↓lr/epochs.  
- **Too slow wall-clock:** ↓n_steps (keep `N_total` via ↑num_envs), ↑minibatch_size.  

---

## 14. Example configs
- **Config A (fast env, short horizon):**  
  `num_envs=16, n_steps=256 → N_total=4096; minibatch=1024; epochs=4; lr=3e-4; ε=0.2`  

- **Config B (MuJoCo-like):**  
  `num_envs=8, n_steps=2048 → N_total=16384; minibatch=2048; epochs=10; lr=3e-4; ε=0.2`  

- **Config C (procgen-ish, high variance):**  
  `num_envs=64, n_steps=256 → N_total=16384; minibatch=2048; epochs=2; lr=5e-4; ε=0.1; entropy=0.01`  

---

## 15. Final reminders
- Keep `N_total` divisible by `minibatch_size`.  
- Favor fixed batch sizes; tune with lr/clip schedules.  
- Evaluate with 3–5 seeds + separate eval env.  
- Change **one knob at a time** (or small grid).  
- Track **samples-to-score**, not just wall-clock.  
