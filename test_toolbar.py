#!/usr/bin/env python3
"""Quick test to verify VisualizationToolbar can be imported and has correct structure."""

import sys

print("Testing VisualizationToolbar implementation...")

# Test 1: Import the module
try:
    from run_play import VisualizationToolbar, CNNFilterActivationViewer
    print("✓ Successfully imported VisualizationToolbar and CNNFilterActivationViewer")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check class attributes
print("\nChecking VisualizationToolbar attributes:")
expected_methods = [
    '__init__',
    '_render',
    'handle_events',
    '_toggle_reward',
    '_toggle_observation',
    '_toggle_filters',
    '_cycle_filter_palette',
    '_cycle_activation_palette',
    'close'
]

for method in expected_methods:
    if hasattr(VisualizationToolbar, method):
        print(f"  ✓ Has method: {method}")
    else:
        print(f"  ✗ Missing method: {method}")
        sys.exit(1)

# Test 3: Check CNNFilterActivationViewer has colormap methods
print("\nChecking CNNFilterActivationViewer colormap methods:")
colormap_methods = ['set_filter_colormap', 'set_activation_colormap']

for method in colormap_methods:
    if hasattr(CNNFilterActivationViewer, method):
        print(f"  ✓ Has method: {method}")
    else:
        print(f"  ✗ Missing method: {method}")
        sys.exit(1)

print("\n✓ All tests passed!")
print("\nToolbar features:")
print("  - Toggle reward plot (R key)")
print("  - Toggle observation viewer (O key)")
print("  - Toggle filter viewer (F key)")
print("  - Cycle filter colormap ([ key)")
print("  - Cycle activation colormap (] key)")
print("  - Close toolbar (Q key)")
print("\nUsage: python run_play.py @last --toolbar")
