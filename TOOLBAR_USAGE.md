# Visualization Toolbar Usage Guide

The visualization toolbar provides an interactive button-based interface for toggling visualizations and changing color palettes during policy playback.

## Quick Start

The toolbar is **enabled by default** when running `run_play.py`:

```bash
# Toolbar enabled by default
python run_play.py @last

# Explicitly disable toolbar
python run_play.py @last --no-toolbar
```

## Interface

The toolbar features a modern button-based UI with:

### Viewer Toggle Buttons
- **■ Reward Plot** - Click to toggle reward plotting on/off (hotkey: R)
- **● Observation Viewer** - Click to toggle observation viewer on/off (hotkey: O)
- **◆ Filter Viewer** - Click to toggle CNN filter viewer on/off (hotkey: F)

Each button shows:
- Icon on the left
- Label in the middle
- Status (ON/OFF) with color indicator
- Hotkey hint [R] [O] [F] on the right

### Colormap Controls
When Filter Viewer is active, colormap controls appear:
- **◀ ▶ Filter: [ColorMapName]** - Click arrows to cycle filter colormaps
- **◀ ▶ Activation: [ColorMapName]** - Click arrows to cycle activation colormaps

### Visual Feedback
- **Hover effect**: Buttons lighten when you hover over them
- **Active state**: Active viewers show with green background and border
- **Status indicators**: ON (green) / OFF (gray)

## Keyboard Shortcuts

All features can also be controlled via keyboard:

- **R** - Toggle Reward Plot
- **O** - Toggle Observation Viewer
- **F** - Toggle Filter Viewer (CNN policies only)
- **[** - Previous Filter Colormap
- **]** - Next Activation Colormap
- **Q** - Close Toolbar

## Color Palettes

### Filter Colormaps (for CNN weights)
Diverging colormaps that show negative (red) to positive (blue) weights:
1. **RdBu** (default) - Red-White-Blue
2. **RdYlBu** - Red-Yellow-Blue
3. **Seismic** - Blue-White-Red
4. **Coolwarm** - Cool-Warm colors

### Activation Colormaps (for CNN activations)
Sequential colormaps for activation intensities:
1. **Viridis** (default) - Purple to Yellow
2. **Plasma** - Purple to Yellow (warmer)
3. **Inferno** - Black to Yellow through red
4. **Magma** - Black to White through purple
5. **Cividis** - Colorblind-friendly blue to yellow

## Features

- **Dynamic Toggle**: Turn visualizations on/off during playback without restarting
- **Live Colormap Switching**: Change color schemes in real-time
- **Status Display**: See which visualizations are active at a glance
- **Lightweight**: Minimal performance impact on playback

## Examples

```bash
# Basic usage with toolbar (default)
python run_play.py @last

# Disable all default visualizations, use only toolbar
python run_play.py @last --no-plot-metrics

# Start with CNN filters enabled
python run_play.py @last --show-cnn-filters

# Disable toolbar completely
python run_play.py @last --no-toolbar --no-plot-metrics
```

## Troubleshooting

**Black window?**
- The bug causing black windows has been fixed. Ensure you're running the latest version.

**Toolbar not responding?**
- Make sure the toolbar window has focus (click on it)
- Try pressing the keys while looking at the toolbar window

**Can't toggle viewers?**
- For Filter Viewer (F key), you need a CNN policy loaded
- For other viewers, they should toggle immediately
