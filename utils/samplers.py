from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler


class MultiPassRandomSampler(Sampler[int]):
    """
    Yields indices for `num_passes` independent random permutations of a dataset.

    This effectively repeats the dataset `num_passes` times per DataLoader epoch,
    shuffling the order for each pass. Useful for PPO where we want K optimization
    epochs over the same rollout within a single Lightning epoch.
    """

    def __init__(
        self,
        data_len: int,
        num_passes: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if data_len <= 0:
            raise ValueError("data_len must be > 0")
        if num_passes <= 0:
            raise ValueError("num_passes must be > 0")
        self.data_len = int(data_len)
        self.num_passes = int(num_passes)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # Produce concatenation of `num_passes` random permutations
        for _ in range(self.num_passes):
            # torch.randperm is efficient and can use an optional generator
            perm = torch.randperm(self.data_len, generator=self.generator)
            for idx in perm.tolist():
                yield int(idx)

    def __len__(self) -> int:
        return self.data_len * self.num_passes
