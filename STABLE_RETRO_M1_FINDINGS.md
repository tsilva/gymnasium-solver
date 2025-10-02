# stable-retro M1 Mac Compatibility Investigation

## Summary
**stable-retro 0.9.5 is NOT compatible with M1 Mac (arm64) architecture.**

## Root Cause
The `stable-retro-0.9.5-cp312-cp312-macosx_11_0_arm64.whl` package on PyPI contains an **x86_64 binary** despite being labeled as an arm64 wheel. This is a packaging bug.

```bash
$ file retro/_retro.cpython-312-darwin.so
retro/_retro.cpython-312-darwin.so: Mach-O 64-bit dynamically linked shared library x86_64
```

## Attempted Fixes

### 1. Reinstalling the wheel
- Direct download and reinstall of the arm64 wheel still installed x86_64 binary
- Conclusion: The wheel itself is mislabeled

### 2. Building from source
- Attempted: `uv pip install --no-binary stable-retro stable-retro`
- Result: Build failed with CMake error
- Error: `add_subdirectory given source "tests" which is not an existing directory`
- Conclusion: Source distribution is also broken

## Solution Implemented
Made stable-retro an **optional dependency**:

1. **pyproject.toml**: Moved to `[project.optional-dependencies]`
   ```toml
   [project.optional-dependencies]
   retro = [
       # NOTE: stable-retro 0.9.5 is BROKEN on M1 Mac
       "stable-retro>=0.9.5",
   ]
   ```

2. **utils/environment.py**: Added fail-fast import check
   ```python
   def _build_env_stable_retro(env_id, obs_type, render_mode, **env_kwargs):
       try:
           import retro
       except ImportError as e:
           raise ImportError(
               f"stable-retro is required for {env_id} but not installed. "
               f"Note: stable-retro 0.9.5 is broken on M1 Mac..."
           ) from e
   ```

3. **Documentation**: Updated README.md and CLAUDE.md

## Workaround
Users can still use stable-retro on M1 Mac by running Python under **Rosetta** (x86_64 emulation):
```bash
arch -x86_64 python train.py Retro/SuperMarioBros-Nes:ppo
```

## Test Script
Created `test_stable_retro.py` to verify compatibility. Run with:
```bash
python test_stable_retro.py
```

## Upstream Issue
This should be reported to the stable-retro maintainers:
- Repository: https://github.com/Farama-Foundation/stable-retro
- Issue: arm64 wheel contains x86_64 binary
- Version: 0.9.5
