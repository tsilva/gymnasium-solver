#!/bin/bash
set -euo pipefail

PACKAGE="stable-retro-apple-silicon==0.9.9.post1"

echo "Installing ${PACKAGE} for Apple Silicon..."

if [[ "${OSTYPE:-}" != darwin* ]]; then
    echo "Error: this installer is for macOS Apple Silicon only."
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: this installer requires arm64 hardware."
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv sync
    PYTHON_CMD=(uv run python)
else
    python3 -m pip install "${PACKAGE}"
    PYTHON_CMD=(python3)
fi

"${PYTHON_CMD[@]}" - <<'PY'
import retro
print(f"retro import OK ({retro.__version__})")
PY

echo "To import ROMs:"
echo "  python3 -m retro.import /path/to/your/roms"
