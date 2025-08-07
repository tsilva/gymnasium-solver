# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning research repository focused on solving OpenAI Gymnasium environments using multiple RL algorithms. The project implements PPO, REINFORCE, and Q-Learning agents with PyTorch Lightning as the training framework.

## Key Commands

### Training
```bash
# Train with default config (CartPole-v1 with PPO)
python train.py

# Train with specific config
python train.py --config CartPole-v1_ppo

# Resume training from checkpoint
python train.py --config CartPole-v1_ppo --resume

# Train with specific algorithm override
python train.py --config CartPole-v1 --algo ppo
```

### Playing/Testing Trained Models
```bash
# Play using latest trained model
python play.py

# Play specific run
python play.py --run-id <run-id>

# Play with different number of episodes
python play.py --run-id <run-id> --episodes 10

# Use stochastic policy instead of deterministic
python play.py --run-id <run-id> --stochastic
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## Architecture Overview

### Core Components

1. **Agents** (`agents/`): Contains base agent class and implementations
   - `base_agent.py`: PyTorch Lightning module base class with training/validation logic
   - `ppo.py`: Proximal Policy Optimization implementation
   - `reinforce.py`: REINFORCE policy gradient implementation  
   - `qlearning.py`: Q-Learning implementation

2. **Configuration System** (`config/`): YAML-based hierarchical configuration
   - `environments/`: Environment-specific configs with algorithm variants
   - Supports inheritance with `inherits` keyword
   - Configs follow pattern: `{env_id}_{algo_id}` (e.g., `CartPole-v1_ppo`)

3. **Utilities** (`utils/`):
   - `config.py`: Configuration loading and management
   - `environment.py`: Environment creation with wrappers
   - `rollouts.py`: Rollout collection for training data
   - `checkpoint.py`: Model checkpointing utilities
   - `run_manager.py`: Training run management and organization
   - `logging.py`: Logging and output capture

4. **Training Infrastructure**:
   - Uses PyTorch Lightning for training orchestration
   - Weights & Biases (wandb) integration for experiment tracking
   - Callbacks for metrics, video logging, checkpointing, hyperparameter scheduling

### Run Management

- Each training run creates a unique directory in `runs/` with timestamp and run ID
- `runs/latest-run` symlink always points to the most recent run
- Run directories contain:
  - `checkpoints/`: Model checkpoints (best_checkpoint.ckpt, last_checkpoint.ckpt)
  - `configs/`: Saved configuration as JSON
  - `logs/`: Training logs
  - `videos/`: Evaluation videos (if enabled)
  - `hyperparam_control/`: Hyperparameter scheduling data

### Configuration System Details

Configs use YAML with inheritance. Key config fields:
- `env_id`: Gymnasium environment identifier
- `algo_id`: Algorithm (ppo, reinforce, qlearning)
- `n_steps`: Steps per rollout collection
- `batch_size`: Training batch size
- `n_epochs`: Training epochs per rollout
- `n_envs`: Number of parallel environments
- `learning_rate`: Learning rate
- `hidden_dims`: Neural network architecture

## Development Guidelines

### Adding New Algorithms
1. Create new class in `agents/` inheriting from `BaseAgent`
2. Implement required methods (marked with `@must_implement`)
3. Add algorithm to `agents/__init__.py` factory function
4. Create environment configs in `config/environments/`

### Adding New Environments
1. Create YAML config file in `config/environments/`
2. Define base config with `__` prefix for inheritance
3. Create algorithm-specific variants
4. Add environment wrappers in `wrappers/` if needed

### Testing
- Use pytest with markers: `unit`, `integration`, `slow`
- Test files follow pattern: `test_*.py`, `*_test.py`, `tests.py`
- Tests are discovered in `tests/`, `learners/`, `utils/` directories

### Debugging Training Issues
- Check `logs/` directory for training logs
- Use wandb dashboard for metrics visualization
- Inspect run directory structure under `runs/`
- Review TODO.md for known issues and planned improvements
- Use `play.py` to visually inspect trained policies

## Important Notes

- Project uses uv for dependency management (see uv.lock)
- PyTorch models are saved with `weights_only=False` for full checkpoint compatibility
- Environment wrappers are automatically applied based on config
- Video logging requires environments with render support
- Some algorithms may have performance regressions (see TODO.md for known issues)