# Tune hyperparameters for speed

## Goal
Given a `<project_id>:<algo>` pair, iteratively adjust the variant's hyperparameters so `train/roll/ep_rew/mean` reaches the environment reward threshold in the least wall-clock time possible while keeping evaluation stochastic.

## Inputs
- Target config identifier (e.g., `CartPole-v1:ppo`)
- Corresponding YAML at `config/environments/<project_id>.yaml`
- Existing run history under `runs/`

## Steps
1. Examine the baseline: open the environment YAML and note key knobs (rollout size, learning rates, entropy/vf coefficients, evaluation cadence, early-stop flags) plus the configured or inferred `reward_threshold`.
2. Launch a reference run: record the start time and run `python train.py <project_id>:<algo> -q`. Do not flip `eval_deterministic`; stochastic evaluation must remain enabled (default).
3. Supervise training: watch stdout or `runs/@last/run.log` for throughput (`train/sys/timing/fps`, `train/sys/timing/time_elapsed`) and when `train/roll/ep_rew/mean` crosses the threshold. Let early stopping finish naturally if configured.
4. Inspect artifacts: after the run, review `runs/@last/metrics.csv` (or the per-run folder) to pinpoint the epoch/timestep and elapsed seconds where `train/roll/ep_rew/mean ≥ reward_threshold`. Capture companion stats like `train/cnt/total_timesteps`, `train/sys/timing/fps`, and `train/roll/fps` to understand bottlenecks.
5. Choose adjustments: based on the metrics (consult `config/metrics.yaml` for definitions), decide which hyperparameters to tweak—e.g., `n_envs`, `n_steps`, `batch_size`, learning-rate schedule, `clip_range`, `ent_coef`, normalization toggles, or evaluation cadence. Target changes that plausibly cut the time-to-threshold without destabilizing training.
6. Edit the variant block inside `config/environments/<project_id>.yaml` to apply the minimal set of adjustments. Keep unrelated settings untouched.
7. Re-run training with the updated config, repeating Steps 3–6. Track each iteration’s wall-clock to threshold and note regressions.
8. When satisfied, summarize which hyperparameters moved, the fastest observed time-to-threshold, and any remaining ideas or risks (e.g., diminishing returns, throughput limits).

## Notes
- Use `runs/@last` for quick access, but archive the exact run id when comparing experiments.
- If early stopping never triggers, consider raising `max_timesteps` slightly while balancing rollout size and learning rate to avoid wasted updates.
- Refer to `config/metrics.yaml` for metric meaning and healthy ranges before reacting to spikes or plateaus.
- Favor few, well-justified edits per iteration so you can attribute gains to specific changes.
- Keep evaluation stochastic; avoid any config changes (including CLI flags) that set `eval_deterministic=True` or otherwise force deterministic policies.
- Track `train/opt/ppo/approx_kl` and `train/opt/ppo/clip_fraction`; if they stay near zero after several epochs the policy updates are too small—raise `policy_lr`, widen `clip_range`, or reduce the effective batch size (smaller `batch_size`, fewer minibatches) to speed convergence while watching the KL warning bounds.
- When early stopping keys off evaluation rewards, calibrate `eval_episodes`: higher counts stabilize metrics, but once learning is stable, lowering the episode count can surface threshold crossings sooner without altering stochastic evaluation.
