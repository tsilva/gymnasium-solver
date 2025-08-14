import pytest
import torch

from utils.samplers import MultiPassRandomSampler


@pytest.mark.unit
def test_multipass_random_sampler_len_and_values_in_range():
    data_len = 8
    passes = 3
    gen = torch.Generator().manual_seed(123)
    sampler = MultiPassRandomSampler(data_len=data_len, num_passes=passes, generator=gen)

    order = list(iter(sampler))
    assert len(order) == data_len * passes
    assert all(0 <= idx < data_len for idx in order)


@pytest.mark.unit
def test_multipass_random_sampler_each_pass_is_permutation():
    data_len = 10
    passes = 4
    gen = torch.Generator().manual_seed(42)
    sampler = MultiPassRandomSampler(data_len=data_len, num_passes=passes, generator=gen)

    order = list(iter(sampler))
    for p in range(passes):
        chunk = order[p * data_len : (p + 1) * data_len]
        assert sorted(chunk) == list(range(data_len))


@pytest.mark.unit
def test_multipass_random_sampler_singleton_dataset_multiple_passes():
    data_len = 1
    passes = 5
    gen = torch.Generator().manual_seed(999)
    sampler = MultiPassRandomSampler(data_len=data_len, num_passes=passes, generator=gen)
    order = list(iter(sampler))
    assert order == [0] * passes


@pytest.mark.unit
def test_multipass_random_sampler_determinism_with_set_epoch():
    data_len = 7
    passes = 2
    sampler = MultiPassRandomSampler(data_len=data_len, num_passes=passes)

    sampler.set_epoch(0)
    order_a = list(iter(sampler))
    sampler.set_epoch(0)
    order_b = list(iter(sampler))
    assert order_a == order_b

    sampler.set_epoch(1)
    order_c = list(iter(sampler))
    assert order_c != order_a


@pytest.mark.unit
def test_multipass_random_sampler_invalid_args():
    with pytest.raises(ValueError):
        MultiPassRandomSampler(data_len=0, num_passes=1)
    with pytest.raises(ValueError):
        MultiPassRandomSampler(data_len=5, num_passes=0)
