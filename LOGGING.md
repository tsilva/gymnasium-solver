# Comprehensive Logging System

This project now includes a comprehensive logging system that captures all stdout/stderr output to both the console and log files. This ensures that no training output is lost and provides a complete record of each training session.

## Features

- **Dual Output**: All output goes to both console (with colors/formatting) and log files (clean text)
- **Automatic Log Management**: Automatic log file rotation with configurable retention
- **Session Tracking**: Each session gets a unique timestamped log file
- **ANSI Code Stripping**: Log files contain clean text without terminal escape sequences
- **Exception Handling**: Proper logging of both successful completions and errors
- **Thread-Safe**: Safe for multi-threaded applications

## Usage

### Training with Logging

The main training script now automatically logs all output:

```bash
# Basic training with default logging
python train.py --config CartPole-v1 --algo ppo

# Training with custom log directory
python train.py --config CartPole-v1 --algo ppo --log-dir my_logs
```

All output including:
- Configuration details
- Training progress
- Metric tables  
- Error messages
- Session timing

Will be saved to log files in the specified directory.

### Playing/Evaluation with Logging

The play script also supports logging:

```bash
# Play with logging enabled
python play.py --config CartPole-v1 --algo ppo --log-dir play_logs
```

### Checkpoint Management with Logging

The checkpoint manager can optionally log its output:

```bash
# List checkpoints with logging
python checkpoint_manager.py --log list

# Clean checkpoints with logging  
python checkpoint_manager.py --log clean ppo CartPole-v1
```

## Log File Structure

Log files are organized as follows:

```
logs/
├── training_20250804_123456_ppo_CartPole-v1.log
├── training_20250804_124012_reinforce_CartPole-v1.log
└── training_20250804_125534_ppo_LunarLander-v2.log
```

Each log file contains:
- Session header with timestamp and configuration
- Complete training output (metrics, progress, etc.)
- Session footer with completion status and timing

## Programmatic Usage

For custom scripts, you can use the logging system directly:

```python
from utils.logging import capture_all_output

# With configuration object
with capture_all_output(config=my_config):
    print("This goes to both console and log file")
    # ... your code here ...

# Without configuration (generic log name)
with capture_all_output(log_dir="my_logs"):
    print("This also gets logged")
    # ... your code here ...
```

## Configuration

The logging system can be configured when setting it up:

```python
from utils.logging import setup_logging

# Configure global settings
setup_logging(
    log_dir="custom_logs",    # Directory for log files
    max_log_files=20          # Keep up to 20 log files
)
```

## Log File Rotation

The system automatically manages log files:
- Keeps a configurable number of recent log files (default: 10)
- Automatically removes older files when the limit is exceeded
- Files are cleaned up based on modification time

## Error Handling

When exceptions occur during logged sessions:
- The exception is logged to the file
- Session end markers indicate whether completion was successful
- Original exceptions are re-raised for proper error handling

## Thread Safety

The logging system is thread-safe and can be used with:
- Multi-threaded training
- Concurrent logging operations
- PyTorch Lightning's multi-process training

## Benefits

1. **Complete Records**: Never lose training output again
2. **Debugging**: Easily review past training sessions
3. **Monitoring**: Track training progress over time  
4. **Compliance**: Maintain audit trails of experiments
5. **Sharing**: Easily share complete training logs with collaborators

## Implementation Details

The logging system uses a `TeeStream` class that:
- Writes to both original stdout/stderr and log files
- Preserves all terminal formatting for console display
- Strips ANSI codes for clean log files
- Handles thread synchronization properly
- Maintains compatibility with existing code

All existing `print()` statements and other stdout/stderr output is automatically captured without requiring code changes.
