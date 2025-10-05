# W&B Sweep Configurations

This directory contains W&B sweep configurations for hyperparameter optimization.

## Sweep Files

- **`cartpole_ppo_grid.yaml`**: Grid search for CartPole PPO parameters
- **`cartpole_ppo_bayes.yaml`**: Bayesian optimization for CartPole PPO
- **`pong_objects_deterministic_ppo_bayes.yaml`**: Bayesian optimization for Pong with object observations

## Sweep Configuration Format

W&B sweep configs are YAML files with the following structure:

```yaml
name: sweep-name                    # Sweep identifier
project: ProjectName                # W&B project (can be overridden)
method: grid|random|bayes          # Optimization method

metric:
  name: train/roll/ep_rew/mean     # Metric to optimize
  goal: maximize|minimize           # Optimization direction

parameters:
  param_name:
    values: [1, 2, 3]              # Grid/random: discrete values
    # OR
    min: 0.0001                     # Random/Bayes: continuous range
    max: 0.01
    distribution: log_uniform       # Optional: sampling distribution

program: train.py                   # Script to run
command:                            # Command template
  - ${env}
  - ${interpreter}
  - ${program}
  - --config_id
  - "CartPole-v1:ppo"
  - --wandb_sweep
```

## Creating a Sweep

### Method 1: Local Sweep Agent

```bash
# Create sweep (returns sweep ID)
wandb sweep config/sweeps/cartpole_ppo_grid.yaml

# Run sweep agent locally
wandb agent <entity>/<project>/<sweep_id>
```

### Method 2: Modal AI (Distributed)

```bash
# Create and launch on Modal in one command
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10

# Or launch existing sweep on Modal
python scripts/sweep_modal.py --sweep-id <sweep_id> --count 20
```

See [scripts/README_MODAL.md](../../scripts/README_MODAL.md) for Modal documentation.

## Sweep Methods

### Grid Search

Exhaustively tries all parameter combinations.

**Best for**:
- Small parameter spaces (< 100 combinations)
- When you need complete coverage
- When compute budget is not a constraint

**Example**:
```yaml
method: grid
parameters:
  learning_rate:
    values: [0.0001, 0.0003, 0.001]
  batch_size:
    values: [32, 64, 128]
# Total runs: 3 × 3 = 9
```

### Random Search

Samples random parameter combinations.

**Best for**:
- Large parameter spaces
- Exploring the space efficiently
- When you have a limited run budget

**Example**:
```yaml
method: random
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
    distribution: log_uniform
  batch_size:
    values: [32, 64, 128, 256]
```

### Bayesian Optimization

Uses previous results to intelligently choose next parameters.

**Best for**:
- Expensive evaluations (long training times)
- Finding global optima
- When you can run many sequential experiments

**Example**:
```yaml
method: bayes
metric:
  name: train/roll/ep_rew/mean
  goal: maximize
parameters:
  learning_rate:
    min: 0.00001
    max: 0.01
    distribution: log_uniform
  clip_range:
    min: 0.1
    max: 0.4
```

## Parameter Types

### Discrete Values

```yaml
batch_size:
  values: [32, 64, 128, 256]
```

### Continuous Ranges

```yaml
learning_rate:
  min: 0.0001
  max: 0.01
  distribution: uniform  # or log_uniform
```

### Distribution Types

- **`uniform`**: Linear sampling between min/max
- **`log_uniform`**: Logarithmic sampling (good for learning rates)
- **`normal`**: Normal distribution (requires mean/std)
- **`q_uniform`**: Quantized uniform (step size)

### Nested Parameters (Schedules)

For schedule parameters (e.g., `policy_lr: {start: 0.001, end: 0.0}`):

```yaml
parameters:
  policy_lr:
    parameters:
      start:
        min: 0.0001
        max: 0.01
        distribution: log_uniform
      end:
        min: 0.0
        max: 0.001
        distribution: uniform
```

## Optimization Strategies

### Quick Exploration (Grid)

1. Define a coarse grid
2. Run sweep
3. Identify promising regions
4. Create refined grid around best region

### Efficient Search (Bayesian)

1. Define parameter ranges
2. Run initial random samples (10-20)
3. Let Bayesian optimization refine
4. Stop when convergence plateaus

### Budget-Constrained (Random)

1. Define parameter ranges
2. Set fixed run count
3. Random sampling explores space efficiently
4. Analyze results to identify patterns

## Modal AI Considerations

When running sweeps on Modal:

### Worker Configuration

Match worker resources to training requirements:

```python
# In modal_sweep_runner.py
@app.function(
    cpu=2.0,        # CPUs per worker
    memory=4096,    # RAM (MB)
    timeout=3600,   # Max seconds per run
)
```

### Parallelism vs. Cost

Trade-off between speed and cost:

```bash
# High parallelism: 50 workers × 1 run each
python scripts/sweep_modal.py --sweep-id <id> --count 50 --runs-per-worker 1

# Low parallelism: 10 workers × 5 runs each
python scripts/sweep_modal.py --sweep-id <id> --count 50 --runs-per-worker 5
```

**High parallelism** (runs-per-worker=1):
- ✓ Fastest overall completion
- ✗ More startup overhead
- ✗ Higher peak cost

**Low parallelism** (runs-per-worker=5+):
- ✓ Less startup overhead
- ✓ Lower peak cost
- ✗ Slower overall completion

### GPU vs. CPU

Most environments work well on CPU:

```bash
# CartPole, MountainCar, etc.
# Use default CPU workers (cheap, fast enough)
python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10
```

For image-based environments (Atari, VizDoom), consider GPU:

```python
# Modify modal_sweep_runner.py
@app.function(
    gpu="T4",  # Cheap GPU option
    # ... other params ...
)
```

### Timeout Settings

Set appropriate timeouts based on expected training time:

| Environment | Typical Time | Recommended Timeout |
|------------|--------------|---------------------|
| CartPole   | 1-5 min      | 600s (10 min)       |
| MountainCar | 5-15 min    | 1800s (30 min)      |
| Atari      | 30-120 min   | 7200s (2 hours)     |

Modify in `modal_sweep_runner.py`:

```python
@app.function(
    timeout=1800,  # 30 minutes
)
```

## Example Workflows

### Quick PPO Tuning (Grid)

```bash
# 1. Create focused grid sweep
cat > config/sweeps/quick_ppo_grid.yaml <<EOF
name: quick-ppo-grid
project: CartPole-v1
method: grid
metric:
  name: train/roll/ep_rew/mean
  goal: maximize
parameters:
  policy_lr:
    values: [0.0001, 0.0003, 0.001]
  clip_range:
    values: [0.1, 0.2]
  n_epochs:
    values: [5, 10]
program: train.py
command:
  - \${env}
  - \${interpreter}
  - \${program}
  - --config_id
  - "CartPole-v1:ppo"
  - --wandb_sweep
EOF

# 2. Launch on Modal (12 combinations)
python scripts/sweep_modal.py config/sweeps/quick_ppo_grid.yaml --count 12
```

### Bayesian Optimization (Long-Running)

```bash
# 1. Create Bayesian sweep (already exists)
# config/sweeps/cartpole_ppo_bayes.yaml

# 2. Launch initial batch
python scripts/sweep_modal.py \
  config/sweeps/cartpole_ppo_bayes.yaml \
  --count 20

# 3. Monitor W&B dashboard
# 4. Launch more runs if needed
python scripts/sweep_modal.py \
  --sweep-id <sweep_id> \
  --count 30
```

## Best Practices

1. **Start small**: Run a few manual experiments before creating sweeps
2. **Use grid for discrete**: Grid search works well for categorical/discrete parameters
3. **Use Bayesian for continuous**: Bayesian optimization excels with continuous ranges
4. **Set sensible ranges**: Don't make parameter ranges too wide
5. **Monitor early**: Check first few runs to catch config errors quickly
6. **Stop early**: Use early stopping to avoid wasting compute on bad configs
7. **Analyze results**: Use W&B's parallel coordinates plot to identify patterns
8. **Iterate**: Refine parameter ranges based on initial results

## See Also

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Modal AI Documentation](../../scripts/README_MODAL.md)
- [Training Guide](../../CLAUDE.md#training)
