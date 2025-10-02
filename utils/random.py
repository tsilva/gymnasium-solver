"""Randomness and reproducibility helpers."""

from typing import Optional
import random

import numpy as np
import torch

# Process-wide torch generator for reproducible shuffles
_global_torch_generator: Optional[torch.Generator] = None

def get_global_torch_generator(seed: Optional[int] = None) -> torch.Generator:
	"""Return a process-wide torch.Generator, optionally seeded once.

	Notes:
	- The first call can pass a seed to initialize determinism. Subsequent calls
		will return the same generator instance and ignore the seed argument.
	- Using a single shared generator keeps shuffles reproducible across users
		without tightly coupling callsites to local seeding logic.
	"""
	global _global_torch_generator
	if _global_torch_generator is None:
			g = torch.Generator()
			if seed is not None:
					g.manual_seed(int(seed))
			_global_torch_generator = g
	return _global_torch_generator

def set_random_seed(seed: int) -> None:
	"""Seed all random number generators for reproducibility.

	Seeds:
	- Python's random module
	- NumPy's random generator
	- PyTorch's RNGs (CPU and CUDA)
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
