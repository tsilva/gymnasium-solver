"""
Comprehensive logging utilities that ensure all stdout output is also logged to files.
"""

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO


class TeeStream:
    """
    A stream wrapper that writes to both the original stream and a log file.
    Preserves all formatting, colors, and special characters.
    """
    
    def __init__(self, original_stream: TextIO, log_file: TextIO):
        self.original_stream = original_stream
        self.log_file = log_file
        self._lock = threading.Lock()
    
    def write(self, data: str) -> int:
        """Write data to both original stream and log file."""
        with self._lock:
            # Write to original stream (with colors/formatting)
            bytes_written = self.original_stream.write(data)
            self.original_stream.flush()
            
            # Write to log file (strip ANSI codes for clean log)
            clean_data = self._strip_ansi_codes(data)
            self.log_file.write(clean_data)
            self.log_file.flush()
            
            return bytes_written
    
    def flush(self):
        """Flush both streams."""
        with self._lock:
            self.original_stream.flush()
            self.log_file.flush()
    
    def isatty(self) -> bool:
        """Return whether the original stream is a TTY."""
        return self.original_stream.isatty()
    
    def fileno(self) -> int:
        """Return the file descriptor of the original stream."""
        return self.original_stream.fileno()
    
    def close(self):
        """Close the log file but not the original stream."""
        with self._lock:
            try:
                if hasattr(self.log_file, 'close') and not self.log_file.closed:
                    self.log_file.close()
            except Exception:
                # Silently ignore errors during close to prevent recursive exceptions
                pass
    
    @staticmethod
    def _strip_ansi_codes(text: str) -> str:
        """Remove ANSI escape sequences from text for clean log files."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def __getattr__(self, name):
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)


class LogFileManager:
    """
    Manages log files with automatic rotation and organization.
    """
    
    def __init__(self, log_dir: str = "logs", max_log_files: int = 10):
        self.log_dir = Path(log_dir)
        self.max_log_files = max_log_files
        self.log_dir.mkdir(exist_ok=True)
        
        # Clean up old log files
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove old log files if we exceed the maximum number."""
        log_files = sorted(self.log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime)
        
        if len(log_files) >= self.max_log_files:
            files_to_remove = log_files[:-self.max_log_files + 1]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except OSError:
                    pass  # Ignore errors if file is already gone
    
    def create_log_file(self, config=None) -> TextIO:
        """
        Create a new log file with a timestamp and optional config info.
        
        Args:
            config: Optional configuration object to include in filename
            
        Returns:
            File handle for the log file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if config and hasattr(config, 'algo_id') and hasattr(config, 'env_id'):
            # Include algorithm and environment in filename
            env_safe = config.env_id.replace("/", "-").replace("\\", "-")
            filename = f"training_{timestamp}_{config.algo_id}_{env_safe}.log"
        else:
            filename = f"training_{timestamp}.log"
        
        log_path = self.log_dir / filename
        
        # Create log file with UTF-8 encoding
        log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
        # Write header with session information
        log_file.write(f"=== Training Session Started ===\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Log file: {log_path}\n")
        
        if config:
            log_file.write(f"Algorithm: {getattr(config, 'algo_id', 'unknown')}\n")
            log_file.write(f"Environment: {getattr(config, 'env_id', 'unknown')}\n")
            log_file.write(f"Seed: {getattr(config, 'seed', 'unknown')}\n")
        
        log_file.write("=" * 50 + "\n\n")
        log_file.flush()
        
        return log_file


class StreamRedirector:
    """
    Context manager for redirecting stdout/stderr to both console and log file.
    """
    
    def __init__(self, log_file: TextIO, redirect_stdout: bool = True, redirect_stderr: bool = True):
        self.log_file = log_file
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        
        self.original_stdout = None
        self.original_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None
    
    def __enter__(self):
        if self.redirect_stdout:
            self.original_stdout = sys.stdout
            self.tee_stdout = TeeStream(self.original_stdout, self.log_file)
            sys.stdout = self.tee_stdout
        
        if self.redirect_stderr:
            self.original_stderr = sys.stderr
            self.tee_stderr = TeeStream(self.original_stderr, self.log_file)
            sys.stderr = self.tee_stderr
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams first
        if self.redirect_stdout and self.original_stdout:
            sys.stdout = self.original_stdout
        
        if self.redirect_stderr and self.original_stderr:
            sys.stderr = self.original_stderr
        
        # Try to write session end marker, but don't fail if it doesn't work
        try:
            if not self.log_file.closed:
                if exc_type is None:
                    self.log_file.write(f"\n\n=== Training Session Completed Successfully ===\n")
                else:
                    self.log_file.write(f"\n\n=== Training Session Ended with Error ===\n")
                    self.log_file.write(f"Error: {exc_type.__name__}: {exc_val}\n")
                
                self.log_file.write(f"End timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write("=" * 50 + "\n")
                self.log_file.flush()
        except Exception:
            # Silently ignore errors during cleanup to prevent recursive exceptions
            pass
        
        # Close TeeStreams after writing session end
        try:
            if self.tee_stdout:
                self.tee_stdout.close()
        except Exception:
            pass
            
        try:
            if self.tee_stderr:
                self.tee_stderr.close()
        except Exception:
            pass


# Global log file manager instance
_log_manager = None


def setup_logging(log_dir: str = "logs", max_log_files: int = 10) -> LogFileManager:
    """
    Set up the global logging system.
    
    Args:
        log_dir: Directory to store log files
        max_log_files: Maximum number of log files to keep
        
    Returns:
        LogFileManager instance
    """
    global _log_manager
    _log_manager = LogFileManager(log_dir, max_log_files)
    return _log_manager


def get_log_manager() -> Optional[LogFileManager]:
    """Get the global log manager instance."""
    return _log_manager


@contextmanager
def capture_all_output(config=None, log_dir: str = "logs", max_log_files: int = 10):
    """
    Context manager that captures all stdout/stderr to both console and log file.
    
    Args:
        config: Optional configuration object for log file naming
        log_dir: Directory to store log files
        max_log_files: Maximum number of log files to keep
        
    Usage:
        with capture_all_output(config):
            # All print statements and other stdout/stderr will be logged
            print("This goes to both console and log file")
            # ... run training ...
    """
    logs_path = Path(log_dir)
    if logs_path.name == "logs":
        # Legacy/generic mode: create timestamped logs under a 'logs' directory
        # Set up log manager if not already done, or if target dir changed
        target_dir = logs_path
        if _log_manager is None or getattr(_log_manager, 'log_dir', None) != target_dir:
            setup_logging(str(target_dir), max_log_files)
        # Create timestamped log file under logs/
        log_file = _log_manager.create_log_file(config)
    else:
        # Run-root mode: write a stable run.log directly in the provided directory
        try:
            logs_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If directory creation fails, fallback to current directory
            logs_path = Path(".")
        stable_path = logs_path / "run.log"
        # Always overwrite on new session start
        log_file = open(stable_path, 'w', encoding='utf-8', buffering=1)
        # Write a header similar to LogFileManager
        log_file.write(f"=== Training Session Started ===\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Log file: {stable_path}\n")
        if config:
            log_file.write(f"Algorithm: {getattr(config, 'algo_id', 'unknown')}\n")
            log_file.write(f"Environment: {getattr(config, 'env_id', 'unknown')}\n")
            log_file.write(f"Seed: {getattr(config, 'seed', 'unknown')}\n")
        log_file.write("=" * 50 + "\n\n")
        log_file.flush()
    
    # Use stream redirector
    with StreamRedirector(log_file, redirect_stdout=True, redirect_stderr=True):
        try:
            yield log_file
        finally:
            if log_file and not log_file.closed:
                log_file.close()


def log_config_details(config, log_file: Optional[TextIO] = None):
    """
    Log detailed configuration information.
    
    Args:
        config: Configuration object
        log_file: Optional log file handle. If None, just prints to stdout.
    """
    from dataclasses import asdict, is_dataclass
    
    output_lines = [
        "\n=== Configuration Details ===",
    ]
    
    if is_dataclass(config):
        config_dict = asdict(config)
        for key, value in sorted(config_dict.items()):
            output_lines.append(f"{key}: {value}")
    else:
        # Handle other config types
        for attr in sorted(dir(config)):
            if not attr.startswith('_'):
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        output_lines.append(f"{attr}: {value}")
                except Exception:
                    pass
    
    output_lines.append("=" * 30)
    output_lines.append("")
    
    output_text = "\n".join(output_lines)
    
    if log_file and not log_file.closed:
        log_file.write(output_text)
        log_file.flush()
    
    # Always print to stdout as well (will be captured by TeeStream if active)
    print(output_text)


def create_algorithm_logger(config) -> Optional[TextIO]:
    """
    Create a dedicated log file for a specific algorithm run.
    
    Args:
        config: Configuration object containing algorithm and environment info
        
    Returns:
        Log file handle or None if setup failed
    """
    try:
        if _log_manager is None:
            setup_logging()
        return _log_manager.create_log_file(config)
    except Exception as e:
        print(f"Warning: Failed to create algorithm log file: {e}")
        return None
