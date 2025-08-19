def prompt_confirm(prompt: str, default: bool = True, quiet: bool = False) -> bool:
    """Prompt user with yes/no. Defaults on empty, non-interactive, EOF, or quiet mode."""
    
    import sys

    yn = "Y/n" if default else "y/N"
    full = f"{prompt} [{yn}]: "

    if quiet:
        print(f"{full}{'Y' if default else 'N'} (quiet)")
        return default

    if not (sys.stdin and sys.stdin.isatty()):
        print(f"{full}{'Y' if default else 'N'} (auto)")
        return default

    try: resp = input(full).strip().lower()
    except EOFError: return default

    return resp.startswith("y") if resp else default
