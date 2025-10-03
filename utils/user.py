import select
import sys


def prompt_confirm(prompt: str, default: bool = True, quiet: bool = False, timeout: float = 10.0) -> bool:
    """Prompt user with yes/no. Defaults on empty, non-interactive, EOF, quiet mode, or timeout.

    Args:
        prompt: Question to ask the user
        default: Default value to return on empty input or timeout
        quiet: If True, skip prompt and return default immediately
        timeout: Seconds to wait before auto-selecting default (0 = no timeout)
    """

    yn = "Y/n" if default else "y/N"
    full = f"{prompt} [{yn}]: "

    if quiet:
        print(f"{full}{'Y' if default else 'N'} (quiet)")
        return default

    if not (sys.stdin and sys.stdin.isatty()):
        print(f"{full}{'Y' if default else 'N'} (auto)")
        return default

    try:
        print(full, end="", flush=True)

        if timeout > 0:
            # Use select to implement timeout on Unix-like systems
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                print(f"{'Y' if default else 'N'} (timeout)")
                return default

        resp = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default

    return resp.startswith("y") if resp else default
