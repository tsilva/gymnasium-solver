"""Randomness and reproducibility helpers."""

from typing import Optional

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
