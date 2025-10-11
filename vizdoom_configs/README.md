# Custom VizDoom Configurations

This directory contains custom VizDoom configuration files for training on original Doom levels.

## Doom E1M1 Setup

To use the `VizDoom-E1M1-v0` environment, you need the original `doom.wad` file from the shareware or registered version of Doom.

### Obtaining doom.wad

You can obtain `doom.wad` legally through:
1. Purchase the game on Steam, GOG, or other platforms
2. Use the shareware version (doom1.wad)
3. Use FreeDoom as a free alternative (freedoom.wad or freedoom1.wad)

### Installation

Place the `doom.wad` file in one of these locations:

1. **VizDoom scenarios directory** (recommended):
   ```bash
   cp doom.wad ~/.local/share/uv/python/*/lib/python*/site-packages/vizdoom/scenarios/
   ```

2. **Custom location via environment variable**:
   ```bash
   export VIZDOOM_SCENARIOS_DIR=/path/to/wad/directory
   ```

3. **Current working directory**:
   ```bash
   cp doom.wad /home/tsilva/repos/tsilva/gymnasium-solver/
   ```

### Using FreeDoom (Free Alternative)

If you don't have the original Doom WAD, you can use FreeDoom:

1. Download FreeDoom from https://freedoom.github.io/
2. Rename `freedoom1.wad` to `doom.wad`
3. Place it in one of the locations above

**Note**: FreeDoom has different graphics/sounds but identical gameplay mechanics.

### Verify Installation

1. Check if doom.wad is found:

```bash
python vizdoom_configs/check_wad.py
```

2. Test that the environment can be loaded:

```bash
python train.py --list-envs E1M1
```

You should see `VizDoom-E1M1-v0:ppo` in the output.

### Training

Once the WAD file is installed:

```bash
python train.py VizDoom-E1M1-v0:ppo
```
