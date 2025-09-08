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


def _color_enabled(stream=None) -> bool:
    """Return True if ANSI colors should be enabled for the given stream."""
    try:
        import os, sys
        s = stream or sys.stdout
        return bool(getattr(s, "isatty", lambda: False)() and os.environ.get("NO_COLOR") is None)
    except Exception:
        return False


def ansi(text: str, *styles: str, enable: bool | None = None) -> str:
    """Wrap text with ANSI styles (supports combos like bold+cyan).

    Styles: red, green, yellow, blue, magenta, cyan, gray, bold, bg_*
    If enable is None, auto-detect using stdout; when False, returns text untouched.
    """
    try:
        if enable is None:
            enable = _color_enabled()
        if not enable or not styles:
            return text
        codes = {
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
            "white": "37",
            "gray": "90",
            "bright_red": "91",
            "bright_green": "92",
            "bright_yellow": "93",
            "bright_blue": "94",
            "bright_magenta": "95",
            "bright_cyan": "96",
            "bright_white": "97",
            "bold": "1",
            "dim": "2",
            # Backgrounds
            "bg_black": "40",
            "bg_red": "41",
            "bg_green": "42",
            "bg_yellow": "43",
            "bg_blue": "44",
            "bg_magenta": "45",
            "bg_cyan": "46",
            "bg_white": "47",
        }
        seq = ";".join(codes[s] for s in styles if s in codes)
        if not seq:
            return text
        return f"\x1b[{seq}m{text}\x1b[0m"
    except Exception:
        return text


def format_kv_line(
    key: str,
    value: object,
    *,
    key_width: int = 20,
    bullet: str = "- ",
    key_color: str = "bright_blue",
    val_color: str = "bright_white",
    enable_color: bool | None = None,
) -> str:
    """Return a single aligned key/value line with colors.

    Example: "-   Key name ..........: Value"
    """
    try:
        s_key = f"{key}:"
        if key_width and key_width > 0:
            s_key = f"{s_key:<{key_width+1}}"  # +1 to include colon in width
        return f"{bullet}{ansi(s_key, key_color, 'bold', enable=enable_color)} {ansi(str(value), val_color, enable=enable_color)}"
    except Exception:
        return f"{bullet}{key}: {value}"


def format_banner(title: str, *, width: int = 60, char: str = "=") -> str:
    """Return a centered, single-line banner like '===== Title ====='.

    - Keeps ASCII by default for broad terminal compatibility.
    - If the title is longer than width, returns the raw title padded with spaces.
    """
    try:
        text = f" {title} "
        if width <= 0:
            return text.strip()
        if len(text) >= width:
            return text
        left = (width - len(text)) // 2
        right = width - len(text) - left
        return f"{char * left}{text}{char * right}"
    except Exception:
        # Be fail-safe — never crash due to formatting
        return f"=== {title} ==="

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
        color = _color_enabled(out if file is not None else None)
        # Fancy banner using heavier char when coloring
        banner_char = "━" if color else "="
        banner = format_banner("Configuration Details", char=banner_char)
        out.write("\n")
        out.write(ansi(banner, "bright_magenta", "bold", enable=color) + "\n")
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
        items = list(sorted(items))
        # Align keys by longest length
        max_k = max((len(k) for k, _ in items), default=0)
        for k, v in items:
            line = format_kv_line(k, v, key_width=max_k, key_color="bright_blue", val_color="bright_white", enable_color=color)
            out.write(line + "\n")
        out.write(ansi(("━" if color else "=") * 60, "bright_magenta", enable=color) + "\n")
        out.flush()
    except Exception:
        # Do not let logging issues crash the program
        pass


def display_config_summary(data_json: dict, *, width: int = 60) -> None:
    """Display structured configuration data in a formatted, colored banner layout.
    
    Args:
        data_json: Dictionary where keys are section titles and values are key-value pairs
        width: Width of the display banner (default: 60)
    """
    use_color = _color_enabled()
    banner_char = "━" if use_color else "="

    def _create_banner(title: str):
        return ansi(format_banner(title, width=width, char=banner_char), "bright_magenta", "bold", enable=use_color)

    def _create_kv_line(key: str, value: str):
        return format_kv_line(key, value, key_width=14, key_color="bright_blue", val_color="bright_white", enable_color=use_color)

    output_lines = []
    for title, data in data_json.items():
        output_lines.append("\n" + _create_banner(title))
        for key, value in data.items():
            output_lines.append(_create_kv_line(key, value))
        output_lines.append(ansi(banner_char * width, "bright_magenta", enable=use_color))
    print("\n".join(output_lines))
