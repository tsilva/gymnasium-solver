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

def get_log_manager() -> Optional[LogFileManager]:
    """Get the global log manager instance."""
    return _log_manager

@contextmanager
def stream_output_to_log(log_file_path: str):
    log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)
    with StreamRedirector(log_file, redirect_stdout=True, redirect_stderr=True):
        try:
            yield log_file
        finally:
            if log_file and not log_file.closed: log_file.close()


@contextmanager
def capture_all_output(config=None, *, log_dir: str = "logs", max_log_files: int = 10):
    """
    Redirect stdout and stderr to both console and a timestamped log file.

    Creates a log directory if needed and rotates old logs based on max_log_files.
    Yields the opened log file handle for optional direct writes.
    """
    # Prepare manager and create a new log file for this session
    manager = LogFileManager(log_dir=log_dir, max_log_files=max_log_files)
    log_file = manager.create_log_file(config)
    # Redirect streams within the context
    with StreamRedirector(log_file, redirect_stdout=True, redirect_stderr=True):
        try:
            yield log_file
        finally:
            try:
                if log_file and not log_file.closed:
                    log_file.close()
            except Exception:
                pass


def log_config_details(config, file: Optional[TextIO] = None) -> None:
    """
    Write human-readable configuration details to the provided file (or stdout).
    Ensures a recognizable header so tests can assert its presence.
    """
    out: TextIO = file or sys.stdout
    try:
        out.write("=== Configuration Details ===\n")
        # Prefer dataclass asdict, fall back to attribute introspection
        items = None
        try:
            from dataclasses import asdict, is_dataclass
            if config is not None and is_dataclass(config):
                items = asdict(config).items()
        except Exception:
            items = None
        if items is None:
            # Collect public, non-callable attributes
            attrs = {}
            if config is not None:
                for name in dir(config):
                    if name.startswith("_"):
                        continue
                    try:
                        value = getattr(config, name)
                    except Exception:
                        continue
                    if callable(value):
                        continue
                    attrs[name] = value
            items = attrs.items()
        for k, v in sorted(items):
            out.write(f"{k}: {v}\n")
        out.flush()
    except Exception:
        # Do not let logging issues crash the program
        pass
