#!/usr/bin/env python3
"""Quick test for CNN filter/activation viewer."""

import torch
import torch.nn as nn
from run_play import CNNFilterActivationViewer


def create_test_cnn_model():
    """Create a simple CNN model for testing."""
    class TestCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.cnn(x)

    return TestCNNModel()


def test_cnn_viewer():
    """Test the CNN filter viewer."""
    print("Creating test CNN model...")
    model = create_test_cnn_model()

    print("Initializing CNN filter viewer...")
    try:
        viewer = CNNFilterActivationViewer(model, update_interval=0.1)
        print(f"✓ Viewer initialized successfully using {'PyQtGraph' if viewer.use_pyqtgraph else 'Matplotlib'}")
        print(f"  Found {len(viewer.conv_layers)} Conv2d layers")

        # Test forward pass to trigger activation capture
        print("\nRunning forward pass...")
        dummy_input = torch.randn(1, 4, 84, 84)  # Batch of 1, 4 channels (frame stack), 84x84
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass completed, output shape: {output.shape}")

        # Update viewer to render activations
        print("\nUpdating viewer...")
        viewer.update()
        print("✓ Viewer updated successfully")

        # Check that activations were captured
        print("\nChecking captured activations...")
        for idx, info in enumerate(viewer.conv_info):
            if info['activation'] is not None:
                act_shape = info['activation'].shape
                print(f"  Layer {idx}: activation shape {act_shape}")
            else:
                print(f"  Layer {idx}: no activation captured")

        print("\nCleaning up...")
        viewer.close()
        print("✓ Viewer closed successfully")

        print("\n✅ All tests passed!")
        return True

    except ValueError as e:
        print(f"✗ ValueError: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cnn_viewer()
    exit(0 if success else 1)
