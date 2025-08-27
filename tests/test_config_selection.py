import pytest

from utils.config import load_config


@pytest.mark.unit
def test_load_config_by_project_defaults_to_first_variant():
    cfg = load_config("LunarLander-v3")
    assert cfg.env_id == "LunarLander-v3"
    # In LunarLander-v3.yaml, the first variant is 'ppo'
    assert cfg.algo_id == "ppo"


@pytest.mark.unit
def test_load_config_by_project_with_explicit_variant():
    cfg = load_config("LunarLander-v3", "reinforce")
    assert cfg.env_id == "LunarLander-v3"
    assert cfg.algo_id == "reinforce"

