# Modal AI Sweep Runner

This directory contains scripts for scaling out W&B hyperparameter sweeps using Modal AI's serverless compute platform.

## Overview

The Modal sweep runner allows you to:
- Scale W&B sweeps across multiple CPU instances in parallel
- Avoid local compute limitations when running large hyperparameter searches
- Pay only for the compute time used (serverless billing)
- Automatically clone your repository on each worker
- Run sweeps with minimal infrastructure setup

## Files

- **`modal_sweep_runner.py`**: Core Modal app that runs sweep agents on cloud CPU instances
- **`sweep_modal.py`**: Convenience wrapper that combines sweep creation and Modal deployment

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install via `uv pip install -e ".[modal]"`
3. **Modal Token**: Run `modal token new` (one-time setup)
4. **W&B API Key**: Set up as Modal secret:
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>
   ```

## Quick Start

### Option 1: One Command (Create + Launch)

```bash
# Create sweep and launch 10 Modal workers
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10
```

### Option 2: Separate Steps

```bash
# 1. Create sweep manually
wandb sweep config/sweeps/cartpole_ppo_grid.yaml
# Output: Created sweep with ID: abc123xyz

# 2. Launch Modal workers
python scripts/sweep_modal.py --sweep-id abc123xyz --count 10
```

## Configuration

### Worker Resources

Each Modal worker runs with:
- **CPU**: 2 cores
- **RAM**: 4GB
- **Timeout**: 1 hour per run
- **Python**: 3.12

Modify these in `modal_sweep_runner.py` if needed:

```python
@app.function(
    cpu=2.0,        # CPUs per worker
    memory=4096,    # RAM in MB
    timeout=3600,   # Max seconds per run
)
```

### Runs Per Worker

Control how many sweep runs each worker executes:

```bash
# 50 total runs, 5 runs per worker = 10 workers launched
python scripts/sweep_modal.py --sweep-id <id> --count 50 --runs-per-worker 5

# 50 total runs, 1 run per worker = 50 workers launched (more parallelism)
python scripts/sweep_modal.py --sweep-id <id> --count 50 --runs-per-worker 1
```

**Trade-offs**:
- **More workers** (runs-per-worker=1): Maximum parallelism, faster overall completion, more startup overhead
- **Fewer workers** (runs-per-worker=5+): Less startup overhead, more sequential execution per worker

### Repository URL

By default, Modal workers clone from `https://github.com/tsilva/gymnasium-solver.git`.

To use a different repository, set the `REPO_URL` environment variable when launching:

```bash
REPO_URL=https://github.com/youruser/your-repo.git \
  python scripts/sweep_modal.py --sweep-id <id> --count 10
```

Or modify the default in `modal_sweep_runner.py`:

```python
repo_url = os.environ.get(
    "REPO_URL", "https://github.com/youruser/your-repo.git"
)
```

## Usage Examples

### Basic Grid Search

```bash
# Create sweep config (config/sweeps/cartpole_ppo_grid.yaml already exists)
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 20
```

### Bayesian Optimization

```bash
# Launch long-running Bayesian sweep (will run indefinitely until stopped)
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_bayes.yaml --count 100
```

### Separate Creation and Launch

```bash
# 1. Create sweep only (get sweep ID)
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --create-only

# 2. Launch workers later
python scripts/sweep_modal.py --sweep-id <sweep_id> --count 30
```

### Custom Entity/Project

```bash
# Override entity and project from command line
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml \
  --entity myusername \
  --project my-project \
  --count 15
```

## Monitoring

### W&B Dashboard

Navigate to your W&B sweep page to monitor:
- Run completion progress
- Metric distributions
- Best parameter combinations
- Parallel execution visualization

```
https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
```

### Modal Logs

View Modal worker logs:

```bash
# List all apps
modal app list

# View logs for specific app
modal app logs wandb-sweep-runner
```

## Cost Estimation

Modal pricing (as of 2024):
- **CPU**: ~$0.0001/second per CPU core
- **RAM**: Included with CPU allocation

Example cost calculation:
- 10 workers × 2 CPUs × 300 seconds = 6000 CPU-seconds
- Cost: 6000 × $0.0001 = $0.60

For current pricing, see [modal.com/pricing](https://modal.com/pricing).

## Troubleshooting

### "Secret not found: wandb-secret"

Create the W&B secret:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

Get your API key from: https://wandb.ai/authorize

### "Could not extract sweep ID from wandb output"

Ensure `wandb` CLI is logged in:

```bash
wandb login
```

Or set `WANDB_API_KEY` environment variable.

### Workers failing during training

Check Modal logs for errors:

```bash
modal app logs wandb-sweep-runner
```

Common issues:
- Missing dependencies (add to `modal_sweep_runner.py` image)
- Repository access (ensure repo is public or add SSH key)
- Timeout too short (increase `timeout` parameter)

### Repository not cloning

Verify the repository URL is accessible:

```bash
git clone <your-repo-url>
```

For private repositories, you'll need to set up SSH keys or deploy tokens in Modal.

## Advanced Usage

### Custom Modal Image

Modify the image definition in `modal_sweep_runner.py` to add dependencies:

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1")  # Add system packages
    .pip_install(
        "your-custom-package>=1.0.0",  # Add Python packages
        # ... existing packages ...
    )
)
```

### GPU Support

To use GPU instances (e.g., for Atari/image-based environments):

```python
@app.function(
    image=image,
    gpu="T4",  # or "A10G", "A100"
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_sweep_agent(...):
    # ... existing code ...
```

Note: GPU instances are significantly more expensive than CPU.

### Different Python Versions

Change Python version in the image:

```python
image = modal.Image.debian_slim(python_version="3.11")  # or "3.10"
```

## Integration with Existing Workflows

### Combining with Local Runs

Run some sweep agents locally while Modal handles the bulk:

```bash
# Terminal 1: Launch Modal workers for 20 runs
python scripts/sweep_modal.py --sweep-id <id> --count 20

# Terminal 2: Run local agent for debugging/monitoring
wandb agent <entity>/<project>/<sweep_id>
```

### CI/CD Integration

Trigger Modal sweeps from CI pipelines:

```yaml
# .github/workflows/sweep.yml
name: W&B Sweep
on:
  workflow_dispatch:
    inputs:
      sweep_config:
        required: true
      count:
        default: '10'

jobs:
  launch-sweep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: modal-labs/modal-action@v1
      - run: |
          python scripts/sweep_modal.py \
            ${{ inputs.sweep_config }} \
            --count ${{ inputs.count }}
```

## See Also

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Modal Documentation](https://modal.com/docs)
- [Sweep Configuration Guide](../config/sweeps/README.md)
