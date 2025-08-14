from typing import Iterator, Optional
import torch
from torch.utils.data import Sampler

class MultiPassRandomSampler(Sampler[int]):
    """
    Yields indices for `num_passes` independent random permutations of a dataset,
    concatenated together. Faster than per-index yielding.
    """
    def __init__(
        self,
        data_len: int,
        num_passes: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if data_len <= 0: raise ValueError("data_len must be > 0")
        if num_passes <= 0: raise ValueError("num_passes must be > 0")
        self.data_len = int(data_len)
        self.num_passes = int(num_passes)
        self.generator = generator or torch.Generator()
        self._base_seed = int(torch.initial_seed())

    def set_epoch(self, epoch: int) -> None:
        # Call this from your training loop (like DistributedSampler does)
        self.generator.manual_seed(self._base_seed + int(epoch))

    def __iter__(self) -> Iterator[int]:
        # Single allocation + single iterator; avoids per-item Python overhead.
        # Using argsort(rand) to produce K permutations in one go.
        scores = torch.rand((self.num_passes, self.data_len), generator=self.generator)
        order = torch.argsort(scores, dim=1).reshape(-1).tolist()
        return iter(order)

    def __len__(self) -> int:
        return self.data_len * self.num_passes