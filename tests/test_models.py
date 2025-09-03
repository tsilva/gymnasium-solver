import pytest
import torch

from utils.models import MLPActorCritic, MLPPolicy


@pytest.mark.unit
def test_policy_only_shapes_and_actions():
    model = MLPPolicy(input_dim=4, hidden_dims=(32,), output_dim=3, activation="relu")
    obs = torch.randn(5, 4)
    dist, value = model(obs)
    # Policy-only returns no value
    assert value is None
    actions = dist.sample()
    logp = dist.log_prob(actions)
    assert actions.shape == (5,)
    assert logp.shape == (5,)


@pytest.mark.unit
def test_actor_critic_shapes_and_values():
    model = MLPActorCritic(input_shape=(4,), hidden_dims=(64,), output_shape=(2,), activation="relu")
    obs = torch.randn(7, 4)
    dist, values = model(obs)
    actions = dist.sample()
    logp = dist.log_prob(actions)
    assert actions.shape == (7,)
    assert logp.shape == (7,)
    assert values.shape == (7,)
