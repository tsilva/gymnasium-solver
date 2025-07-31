#!/usr/bin/env python3

import wandb
import numpy as np
import torch

# Simple test to verify histogram logging works
def test_histogram_logging():
    # Initialize wandb
    run = wandb.init(project="test-histogram", name="debug-test")
    
    try:
        # Create some sample action data
        action_data = np.random.randint(0, 4, size=1000)  # Simulate CartPole actions (0-3)
        
        print(f"Action data shape: {action_data.shape}")
        print(f"Action data type: {type(action_data)}")
        print(f"Action data sample: {action_data[:10]}")
        print(f"Action data unique values: {np.unique(action_data)}")
        
        # Test direct wandb.Histogram logging
        print("Logging histogram directly...")
        wandb.log({
            "test/action_distribution_direct": wandb.Histogram(action_data)
        }, step=1)
        
        # Test logging through logger.experiment (simulate PyTorch Lightning)
        print("Logging histogram through logger.experiment...")
        run.log({
            "test/action_distribution_experiment": wandb.Histogram(action_data)
        }, step=2)
        
        # Test with different data types
        action_data_list = action_data.tolist()
        print("Logging histogram with list data...")
        wandb.log({
            "test/action_distribution_list": wandb.Histogram(action_data_list)
        }, step=3)
        
        # Test with torch tensor
        action_data_torch = torch.tensor(action_data)
        print("Logging histogram with torch tensor...")
        wandb.log({
            "test/action_distribution_torch": wandb.Histogram(action_data_torch.numpy())
        }, step=4)
        
        print("All histograms logged successfully!")
        print(f"WandB run URL: {run.url}")
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    test_histogram_logging()
