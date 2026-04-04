import pytest

from utils.config import load_config


@pytest.mark.unit
def test_load_config_requires_explicit_variant():
    with pytest.raises(ValueError, match="variant_id is required"):
        load_config("LunarLander-v3")


@pytest.mark.unit
def test_load_config_rejects_legacy_combined_id():
    with pytest.raises(ValueError, match="variant_id is required"):
        load_config("CartPole-v1_ppo")


@pytest.mark.unit
def test_load_config_with_explicit_variant():
    cfg = load_config("LunarLander-v3", "ppo")
    assert cfg.env_id == "LunarLander-v3"
    assert cfg.algo_id == "ppo"


@pytest.mark.unit
def test_load_config_accepts_raw_env_id_alias_with_explicit_variant():
    cfg = load_config("ALE/Pong-v5", "objects_ppo")
    assert cfg.env_id == "ALE/Pong-v5"
    assert cfg.algo_id == "ppo"
