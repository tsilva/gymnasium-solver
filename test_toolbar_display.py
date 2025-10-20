#!/usr/bin/env python3
"""Standalone test to verify the visualization toolbar renders properly."""

import sys
import time

print("Testing VisualizationToolbar display...")
print("This will open a pygame window for 5 seconds to verify rendering.")
print("You should see a dark gray window with white text showing keyboard controls.\n")

try:
    import pygame
except ImportError:
    print("✗ pygame not installed. Install with: pip install pygame")
    sys.exit(1)

try:
    # Create a minimal config mock
    class MockConfig:
        def __init__(self):
            self.spec = {}

    config = MockConfig()

    # Import and create toolbar
    from run_play import VisualizationToolbar

    print("Creating toolbar...")
    toolbar = VisualizationToolbar(config, policy_model=None, env=None)

    print("✓ Toolbar created successfully!")
    print("Window should now be visible with:")
    print("  - Title: 'Visualization Toolbar'")
    print("  - Keyboard controls listed")
    print("  - Active visualizations status (all OFF)")
    print("\nWindow will close automatically in 5 seconds...")
    print("Or press Q to close immediately.\n")

    # Keep window open for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5.0:
        if not toolbar.is_open:
            break

        # Handle events to keep window responsive
        toolbar.handle_events()
        time.sleep(0.1)

    # Clean up
    if toolbar.is_open:
        toolbar.close()

    print("\n✓ Test completed successfully!")
    print("If you saw a dark gray window with white text, the toolbar is working correctly.")

except Exception as e:
    print(f"\n✗ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
