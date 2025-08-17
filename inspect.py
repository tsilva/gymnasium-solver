#!/usr/bin/env python3
"""
Shim for the Inspector app.

- When executed as a script: launches the Gradio inspector UI.
- When imported as a module (e.g., by libraries importing Python's stdlib `inspect`):
  dynamically loads the real stdlib inspect module and exposes it, avoiding breakage.
"""

import sys

if __name__ != "__main__":
    # Provide the real stdlib inspect when imported as a module
    import os
    import sysconfig
    import importlib.util

    stdlib_dir = sysconfig.get_paths().get("stdlib")
    if stdlib_dir:
        real_path = os.path.join(stdlib_dir, "inspect.py")
        spec = importlib.util.spec_from_file_location("_stdlib_inspect", real_path)
        if spec and spec.loader:
            real_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(real_mod)
            # Update our module dict with the real inspect's attributes
            globals().update(real_mod.__dict__)
            # Ensure imports receive this module instance
            sys.modules[__name__] = sys.modules.get(__name__, real_mod)
else:
    # Run the app launcher when executed directly
    try:
        from inspector_app import main as _main
    except Exception as e:
        print(f"Failed to start inspector app: {e}")
        sys.exit(1)
    _main()
