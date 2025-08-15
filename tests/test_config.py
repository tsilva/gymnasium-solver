import json
from pathlib import Path
import pytest

from utils.config import Config


@pytest.mark.unit
def test_config_parse_schedules_and_legacy_normalize():
    cfg = Config._load_from_environment_config({
        "env_id": "CartPole-v1",
        "algo_id": "ppo",
        "n_steps": 32,
        "batch_size": 16,
        "learning_rate": "lin_3e-4",
        "normalize": True,
        "hidden_dims": [64, 64],
    })
    assert cfg.policy_lr == pytest.approx(3e-4)
    assert cfg.learning_rate_schedule == "linear"
    assert cfg.normalize_obs is True and cfg.normalize_reward is True
    assert isinstance(cfg.hidden_dims, tuple)


@pytest.mark.unit
def test_config_environment_format_with_inheritance():
    # Build an in-memory environment-centric config with inheritance
    all_configs = {
        "base": {
            "env_id": "CartPole-v1",
            "algo_id": "ppo",
            "n_steps": 64,
            "batch_size": 32,
            "gamma": 0.98,
        },
        "child": {
            "inherits": "base",
            "n_steps": 128,
            "clip_range": 0.2,
        },
    }
    cfg = Config._load_from_environment_config(all_configs["child"], all_configs)
    assert cfg.env_id == "CartPole-v1"
    assert cfg.algo_id == "ppo"
    assert cfg.n_steps == 128  # override applied
    assert cfg.batch_size == 32
    assert cfg.gamma == pytest.approx(0.98)
    assert cfg.clip_range == pytest.approx(0.2)


@pytest.mark.unit
def test_config_legacy_format_with_inheritance(tmp_path: Path):
    # Build legacy hyperparams folder
    hyper = tmp_path / "hyperparams"
    hyper.mkdir(parents=True, exist_ok=True)
    ppo_yaml = hyper / "ppo.yaml"
    ppo_content = {
        "base_cartpole": {
            "env_id": "CartPole-v1",
            "n_steps": 32,
            "batch_size": 16,
            "gamma": 0.99,
        },
        "cartpole_large": {
            "inherit_from": "base_cartpole",
            "n_steps": 128,
        },
    }
    # JSON is valid YAML; safe_load can parse it
    ppo_yaml.write_text(json.dumps(ppo_content))

    # When algo_id provided and config_id is present in file
    cfg = Config._load_from_legacy_config("cartpole_large", "ppo", hyper)
    assert cfg.env_id == "CartPole-v1"
    assert cfg.n_steps == 128
    assert cfg.batch_size == 16

    # When config_id equals env_id, should find matching config once
    cfg2 = Config._load_from_legacy_config("CartPole-v1", "ppo", hyper)
    assert cfg2.env_id == "CartPole-v1"


@pytest.mark.unit
def test_config_validate_errors():
    # Invalid learning rate
    bad = Config(env_id="CartPole-v1", algo_id="ppo", n_steps=32, batch_size=32, policy_lr=0.0)
    with pytest.raises(ValueError):
        bad.validate()

    # Invalid gamma
    bad2 = Config(env_id="CartPole-v1", algo_id="ppo", n_steps=32, batch_size=32, gamma=1.5)
    with pytest.raises(ValueError):
        bad2.validate()
