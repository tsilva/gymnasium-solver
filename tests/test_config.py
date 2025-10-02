import json
from pathlib import Path

import pytest

from utils.config import Config


@pytest.mark.unit
def test_config_parse_schedules_with_dict_syntax():
    cfg = Config.build_from_dict({
        "env_id": "CartPole-v1",
        "algo_id": "ppo",
        "n_steps": 32,
        "n_envs": 4,
        "batch_size": 16,
        "max_timesteps": 1000,
        "policy_lr": {"start": 3e-4, "end": 0.0},
        "hidden_dims": [64, 64],
    })
    assert cfg.policy_lr == pytest.approx(3e-4)
    assert cfg.policy_lr_schedule == "linear"
    assert cfg.policy_lr_schedule_start_value == pytest.approx(3e-4)
    assert cfg.policy_lr_schedule_end_value == pytest.approx(0.0)
    assert cfg.policy_lr_schedule_start == pytest.approx(0.0)
    assert cfg.policy_lr_schedule_end == pytest.approx(1.0)


@pytest.mark.unit
def test_config_schedule_fraction_without_max_timesteps_errors():
    with pytest.raises(ValueError):
        Config.build_from_dict(
            {
                "algo_id": "ppo",
                "env_id": "CartPole-v1",
                "n_steps": 32,
                "batch_size": 32,
                "policy_lr": 3e-4,
                "policy_lr_schedule": "linear",
                "policy_lr_schedule_start": 0.0,
                "policy_lr_schedule_end": 0.5,
            }
        )






@pytest.mark.unit
def test_config_validate_errors():
    # Invalid learning rate
    with pytest.raises(ValueError):
        Config.build_from_dict({
            "env_id": "CartPole-v1",
            "algo_id": "ppo",
            "n_steps": 32,
            "n_envs": 4,
            "batch_size": 32,
            "policy_lr": 0.0
        })

    # Invalid gamma
    with pytest.raises(ValueError):
        Config.build_from_dict({
            "env_id": "CartPole-v1",
            "algo_id": "ppo",
            "n_steps": 32,
            "n_envs": 4,
            "batch_size": 32,
            "gamma": 1.5
        })


@pytest.mark.unit
def test_config_schedule_dict_syntax():
    """Test new dict-based schedule syntax"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_timesteps": 10000,
        "ent_coef": {"start": 0.02, "end": 0.001},
        "policy_lr": {"start": 0.003, "end": 0.0001, "from": 0.0, "to": 0.8},
        "clip_range": {"start": 0.2, "end": 0.05, "schedule": "linear"},
    })

    # Check base values are set to start values
    assert cfg.ent_coef == pytest.approx(0.02)
    assert cfg.policy_lr == pytest.approx(0.003)
    assert cfg.clip_range == pytest.approx(0.2)

    # Check schedule attributes are populated correctly
    assert cfg.ent_coef_schedule == "linear"
    assert cfg.ent_coef_schedule_start_value == pytest.approx(0.02)
    assert cfg.ent_coef_schedule_end_value == pytest.approx(0.001)
    assert cfg.ent_coef_schedule_start == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_end == pytest.approx(1.0)

    assert cfg.policy_lr_schedule == "linear"
    assert cfg.policy_lr_schedule_start_value == pytest.approx(0.003)
    assert cfg.policy_lr_schedule_end_value == pytest.approx(0.0001)
    assert cfg.policy_lr_schedule_start == pytest.approx(0.0)
    assert cfg.policy_lr_schedule_end == pytest.approx(0.8)

    assert cfg.clip_range_schedule == "linear"
    assert cfg.clip_range_schedule_start_value == pytest.approx(0.2)
    assert cfg.clip_range_schedule_end_value == pytest.approx(0.05)


@pytest.mark.unit
def test_config_schedule_dict_minimal_syntax():
    """Test minimal dict schedule with just start and end"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_timesteps": 10000,
        "ent_coef": {"start": 0.01, "end": 0.0},
    })

    assert cfg.ent_coef == pytest.approx(0.01)
    assert cfg.ent_coef_schedule == "linear"
    assert cfg.ent_coef_schedule_start_value == pytest.approx(0.01)
    assert cfg.ent_coef_schedule_end_value == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_start == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_end == pytest.approx(1.0)
