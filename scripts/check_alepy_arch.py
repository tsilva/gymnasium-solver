import glob
import os
import platform
import subprocess

import ale_py


def check_ale_arch():
    ale_dir = os.path.dirname(ale_py.__file__)

    # Look for any compiled .so file in ale_py directory
    so_files = glob.glob(os.path.join(ale_dir, "*.so"))
    if not so_files:
        print("❌ No compiled ALE C++ library found (.so missing).")
        print(f"Current ale_py is running from: {ale_py.__file__}")
        return

    so_path = so_files[0]  # take the first match (should only be one)

    # Check architecture
    result = subprocess.run(["file", so_path], capture_output=True, text=True)
    print("Python architecture:", platform.machine())
    print("ALE C++ library path:", so_path)
    print("ALE C++ library arch info:", result.stdout.strip())

    if "arm64" in result.stdout:
        print("✅ ALE build is ARM64 (Apple Silicon optimized).")
    elif "x86_64" in result.stdout:
        print("⚠️ ALE build is Intel x86_64 (running under Rosetta). Consider rebuilding with Clang for ARM64.")
    else:
        print("❓ Unknown architecture detected.")

if __name__ == "__main__":
    check_ale_arch()