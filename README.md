## gymnasium-solver 🤖🏋️

Fast, practical reinforcement learning on Gymnasium. Train PPO/REINFORCE agents with config-first workflows, vectorized environments, videos, a Gradio run inspector, and one-command publishing to the Hugging Face Hub.

### ⚠️ Warning
This project is currently for self-education purposes only. I'm doing a lot of vibe coding: I quickly vibe-code features, then review the code and chisel out the AI slop. At any point the codebase may be ugly and buggy. Please don't assume it's a "real" project until I make the first release. From that point on, I'll start working with branches and a more stable workflow.

### ✨ Highlights
- **Algorithms** 🧠: PPO, REINFORCE
- **Config-first** ⚙️: concise YAML configs with inheritance and linear schedules (e.g., `lin_0.001`)
- **Vectorized envs** ⚡: Dummy/Subproc, frame stacking, obs/reward normalization
- **Atari-ready** 🕹️: ALE with `obs_type` rgb/ram/objects (via [Gymnasium](https://gymnasium.farama.org) and [OCAtari](https://github.com/Kautenja/oc-atari))
- **Retro-ready** 🎮: Classic console games via [stable-retro](https://github.com/Farama-Foundation/stable-retro) (e.g., `Retro/SuperMarioBros-Nes`)
- **Wrappers registry** 🧰: plug-in env wrappers by name
- **Great UX** ✨: curated `runs/` folders, auto `@latest-run` link, video capture
- **Inspector UI** 🔎: step-by-step episode browser (Gradio)
- **Hub publishing** 📤: push run artifacts and preview video to [Hugging Face Hub](https://huggingface.co)

### 📦 Install
- Using uv (recommended):
```bash
pipx install uv  # or: pip install uv
uv sync
```
- Or with pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 🚀 Quickstart
- **Train** 🏃 (pass config either positionally or via `--config_id`; defaults to `Bandit-v0:ppo` when omitted):
```bash
# Positional config (shorthand)
python train.py CartPole-v1:ppo -q
python train.py CartPole-v1:reinforce -q

# Or with explicit flag
python train.py --config_id "CartPole-v1:ppo" -q
python train.py --config_id "CartPole-v1:reinforce" -q

# Uses train.py default (Bandit-v0:ppo) when --config_id is omitted
python train.py -q

# Override max timesteps without editing YAML
python train.py CartPole-v1:ppo --max-steps 5000
```
- **Play a trained policy** 🎮 (auto-loads best/last checkpoint from a run):
```bash
python play.py --run-id @latest-run --episodes 5
```
- **Inspect a run (UI)** 🔍:
```bash
python inspector.py --run-id @latest-run --port 7860 --host 127.0.0.1
```

### ⚙️ Configs (YAML)
Configs live in `config/environments/*.yaml`. New style puts base fields at the top and per-algorithm variants under their own key. Linear schedules like `lin_0.001` are parsed automatically. The loader selects an algo-specific config subclass based on `algo_id` (e.g., `PPOConfig`, `REINFORCEConfig`).

```yaml
# New per-file style
# Tip: omit project_id to default to the filename (e.g., CartPole-v1)
env_id: CartPole-v1
eval_episodes: 10

ppo:
  algo_id: ppo
  n_envs: 8
  max_timesteps: 1e5
  n_steps: 32
  batch_size: 256
  policy_lr: lin_0.001   # linear schedule from 0.001 → 0
  clip_range:   lin_0.2
  env_wrappers:
    - { id: CartPoleV1_RewardShaper, angle_reward_scale: 1.0 }
```

Selection:
- Programmatic (Python): `load_config("CartPole-v1", "ppo")`; pass the environment id and variant explicitly because the compact `"CartPole-v1_ppo"` form is no longer supported.
- CLI (train.py): pass config as positional `CartPole-v1:ppo` or flag `--config_id "CartPole-v1:ppo"` (colon, not underscore). The CLI enforces providing a variant.
Callers must always supply a variant; the loader no longer falls back to the first block in the YAML file.
The loader remains compatible with the legacy multi-block format for a transitional period. When `project_id` is omitted in environment YAMLs, it is inferred from the file name.

Key fields: `env_id`, `algo_id`, `n_envs`, `n_steps`, `batch_size`, `max_timesteps`, `policy` (`mlp|cnn`), `hidden_dims`, `obs_type` (`rgb|ram|objects` for ALE), `optimizer` (`adamw` default; supports `adam|adamw|sgd`).

Batch size can be either an absolute integer or a fraction in (0, 1]. If fractional, it is resolved as `batch_size = floor(n_envs * n_steps * fraction)`, with a minimum of 1.

- Advantage normalization: set `normalize_advantages` to `rollout` (normalize once per rollout, SB3-style), `batch` (normalize per mini-batch), or `off`.

REINFORCE options: set `returns_type` to control Monte Carlo returns used for scaling log-probs: `mc:rtg` (reward-to-go; default) or `mc:episode` (classic vanilla: make return constant across each episode segment).

VizDoom support: set `env_id` to `VizDoom-DeadlyCorridor-v0`. Requires `pip install vizdoom` and access to `deadly_corridor.cfg`/`deadly_corridor.wad` (auto-discovered from the installed package, or set `VIZDOOM_SCENARIOS_DIR` or `env_kwargs.config_path`).

### 🧰 Environment wrappers
Register-by-name wrappers via `EnvWrapperRegistry` (see `gym_wrappers/__init__.py`). Available IDs:
- `PixelObservationWrapper`
- `DiscreteEncoder` (encoding: 'array' | 'binary' | 'onehot')
- `PongV5_FeatureExtractor`, `PongV5_RewardShaper`
- `MountainCarV0_RewardShaper`, `CartPoleV1_RewardShaper`, `VizDoom_RewardShaper`

Use in YAML:
```yaml
env_wrappers:
  - { id: PixelObservationWrapper, pixels_only: true }
```

### 🎥 Runs, checkpoints, and videos
- 📁 Each training creates `runs/<id>/` with `config.json`, `checkpoints/*.ckpt`, `logs/`, and `videos/`
- 🔗 `runs/@latest-run` symlink points to the most recent run
- 🏷️ Best/last checkpoints: `best.ckpt`, `last.ckpt` (auto-detected by `play.py` and the inspector)
- 📈 Metrics: prints and logs `train/*` and `eval/*` including `ep_rew_mean` and running best as `ep_rew_best` (highlighted in blue; rules configurable in `config/metrics.yaml`). Metrics that fall outside configured bounds (`min`/`max` in `config/metrics.yaml`) are highlighted in yellow and emit a console warning. The console table also shows an inline ASCII sparkline (e.g., `█▇▇▆▅▄▃▂▁`) per numeric metric to visualize recent trends.

### 📤 Publish to Hugging Face Hub
Authenticate once (`huggingface-cli login`) or set `HF_TOKEN`, then:
```bash
python publish.py                 # publish latest run
python publish.py --run-id <ID>   # publish a specific run
python publish.py --repo user/repo --private
```
Uploads run artifacts under `artifacts/` and attaches a preview video when found.

### 🗂️ Project layout
```
agents/            # PPO, REINFORCE
loggers/           # custom Lightning/CSV/console loggers
utils/             # config, env, models, rollouts, helpers
gym_wrappers/      # registry + wrappers (feature extractors, reward shaping, pixels)
trainer_callbacks/ # logging, early stopping, checkpointing, hyperparam sync, videos
config/            # environment YAML configs
runs/              # training outputs (checkpoints, videos, logs, config)
```

### 🌀 W&B Sweeps
- Train under a sweep: set your sweep `program` to call `python train.py --config_id "<env>:<variant>"` (e.g., `CartPole-v1:ppo`).
- The script auto-detects W&B Agent via `WANDB_SWEEP_ID` and merges `wandb.config` into the main config before training. You can also force this behavior with `--wandb_sweep`.
- Parameter names map 1:1 to config fields (e.g., `n_envs`, `n_steps`, `batch_size`, `policy_lr`, `clip_range`, `gamma`, `gae_lambda`, `ent_coef`, `vf_coef`, `max_timesteps`, etc.). Linear schedules like `lin_0.001` are supported for sweep values.
- Fractional `batch_size` in (0, 1] is resolved as a fraction of `n_envs * n_steps` after overrides are applied.

Example sweep specs:
- Grid: `config/sweeps/cartpole_ppo_grid.yaml`
- Bayesian: `config/sweeps/cartpole_ppo_bayes.yaml`

Usage:
```bash
wandb sweep config/sweeps/cartpole_ppo_grid.yaml
wandb agent <entity>/<project>/<sweep_id>
```

### 📊 W&B Workspace
Training auto-creates/updates a project workspace at the end, default-selects the current run across panels, and prints the URL (uses the active run's entity/project; workspace name: "<project> View").

Create or preview a default W&B dashboard manually (idempotent):
```bash
# Ensure dependencies
uv sync  # or: pip install -e . && pip install wandb-workspaces

# Push a workspace (reads WANDB_ENTITY/WANDB_PROJECT by default).
# Safe by default: will not overwrite an existing workspace unless --overwrite is passed.
python scripts/setup_wandb_dashboard.py --entity <team-or-user> --project <project-name>

# Default-select the most recent run across panels
python scripts/setup_wandb_dashboard.py --entity <team-or-user> --project <project-name> --select-latest

# Dry-run to preview JSON without pushing
python scripts/setup_wandb_dashboard.py --dry-run
```
Requires a logged-in W&B session (`wandb login`) or `WANDB_API_KEY` set.

- Notes:
  - The script won’t overwrite an existing workspace by default; it prints that it already exists. Use `--overwrite` to update the existing layout.
  - Use `--key-panels-per-section N` to control how many “Key Metrics” panels appear per section.
  - Default workspace name is "<project> View". Override with `--name`.

### 🔇 Silencing logs
To suppress creation of session log files and banner/config dumps to `logs/` (useful for ad‑hoc play sessions), set:
```
export VIBES_DISABLE_SESSION_LOGS=1
```
To silence verbose checkpoint load prints, set:
```
export VIBES_QUIET=1   # or: VIBES_DISABLE_CHECKPOINT_LOGS=1
```

### 🧪 Tests
```bash
pytest -q
```

### 🧑‍💻 Developer tasks
- `!TASK: audit_bugs` - Find and document functional defects with precise remediation plans
- `!TASK: audit_static_analysis` - Identify latent defects using lightweight static checks
- `!TASK: cleanup_dead_code` - Find and remove unused modules, functions, and branches
- `!TASK: cleanup_dry_file` - Reduce duplication within a single module by extracting shared behavior
- `!TASK: audit_dryness` - Identify copy-paste and near-duplicate logic across the codebase
- `!TASK: audit_encapsulation` - Find places where shared behavior can be encapsulated
- `!TASK: audit_concerns` - Find separation of concerns issues across layers or modules
- `!TASK: audit_config_consistency` - Validate inconsistencies across configuration files and registries
- `!TASK: audit_dependency_health` - Assess and document the state of runtime and development dependencies
- `!TASK: tune_hyperparams` - Iteratively adjust hyperparameters for fastest convergence
- `!TASK: update_docs_architecture` - Ensure Architecture Guide stays accurate and useful
- `!TASK: update_docs_coding_from_diff` - Infer coding preferences from uncommitted changes
- `!TASK: update_docs_readme` - Keep README concise, accurate, and aligned with current features
- `!TASK: audit_tests` - Exercise all test suites and document failures or flakiness
- `!TASK: test_upgrade` - Strengthen test suite by adding tests for high-risk modules
- `!TASK: cleanup_imports` - Streamline Python imports to reflect actual usage and follow conventions

### 📄 License
MIT
