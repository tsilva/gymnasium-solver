import torch


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, length: int):
        self._len = length
    def __len__(self) -> int:
        return self._len
    def __getitem__(self, idx: int) -> int:
        return idx