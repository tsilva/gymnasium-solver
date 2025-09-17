import math
import pytest
import torch

from utils.policy_factory import build_policy


@pytest.mark.unit
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
@pytest.mark.parametrize("hidden_dims", [
    (16,), (32, 16),
])
def test_mlp_policy_uniform_init(activation, hidden_dims):
    """
    Policies created via the factory should start unbiased: given zero observations,
    their action distribution should be uniform (equal probabilities across actions).
    This guards against biased exploration at initialization time.
    """
    model = build_policy(
        "mlp",
        input_shape=(4,),
        hidden_dims=hidden_dims,
        output_shape=(5,),
        activation=activation,
    )

    batch_size = 3
    obs = torch.zeros(batch_size, 4)
    dist, value = model(obs)

    # Policy-only models return no value head
    assert value is None

    probs = dist.probs  # (N, A)
    assert probs.shape == (batch_size, 5)

    # Each row should be uniform: equal probs across actions and sum to 1
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    # Entropy equals log(A) for a uniform categorical
    A = 5
    expected_entropy = math.log(A)
    entropy = dist.entropy()
    assert torch.allclose(entropy, torch.full_like(entropy, expected_entropy), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
@pytest.mark.parametrize("hidden_dims", [
    (16,), (32, 16),
])
def test_mlp_actor_critic_uniform_init(activation, hidden_dims):
    """
    Actor-critic policies from the factory should also be unbiased at start.
    With zero observations, the policy's distribution should be uniform and
    the value head should output zeros.
    """
    model = build_policy(
        "mlp_actorcritic",
        input_shape=(4,),
        hidden_dims=hidden_dims,
        output_shape=(6,),
        activation=activation,
    )

    batch_size = 4
    obs = torch.zeros(batch_size, 4)
    dist, values = model(obs)

    probs = dist.probs  # (N, A)
    assert probs.shape == (batch_size, 6)
    assert values.shape == (batch_size,)

    # Policy should be uniform across actions
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    # Value head should start at zero for zero inputs (zero bias)
    assert torch.allclose(values, torch.zeros_like(values), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("policy_type", ["cnn", "foo"])
def test_build_policy_unsupported_type(policy_type):
    with pytest.raises(KeyError):
        build_policy(
            policy_type,
            input_shape=(4,),
            hidden_dims=(32,),
            output_shape=(3,),
            activation="relu",
        )
