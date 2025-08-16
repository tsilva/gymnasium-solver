import pytest
import torch

from utils.models import ActorCritic, PolicyOnly


@pytest.mark.unit
def test_policy_only_shapes_and_actions():
    model = PolicyOnly(state_dim=4, action_dim=3)
    obs = torch.randn(5, 4)
    a, logp, v = model.act(obs)
    assert a.shape == (5,)
    assert logp.shape == (5,)
    assert v.shape == (5,)


@pytest.mark.unit
def test_actor_critic_shapes_and_values():
    model = ActorCritic(state_dim=4, action_dim=2)
    obs = torch.randn(7, 4)
    a, logp, v = model.act(obs)
    assert a.shape == (7,)
    assert logp.shape == (7,)
    assert v.shape == (7,)

    vals = model.predict_values(obs)
    assert vals.shape == (7,)
