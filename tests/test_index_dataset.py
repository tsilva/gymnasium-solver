import pytest

from utils.datasets import IndexDataset


@pytest.mark.unit
def test_index_dataset_len_and_getitem_basic():
    ds = IndexDataset(length=7)
    assert len(ds) == 7
    for i in range(len(ds)):
        assert ds[i] == i


@pytest.mark.unit
def test_index_dataset_with_dataloader_integration():
    ds = IndexDataset(length=5)
    indices = list(range(len(ds)))
    values = [ds[i] for i in indices]
    assert values == indices
