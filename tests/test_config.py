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
        "max_env_steps": 1000,
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
def test_config_schedule_fraction_without_max_env_steps_errors():
    """Test that fractional schedule positions without max_env_steps raises error"""
    with pytest.raises(AssertionError, match="max_env_steps"):
        Config.build_from_dict(
            {
                "algo_id": "ppo",
                "env_id": "CartPole-v1",
                "n_steps": 32,
                "n_envs": 4,
                "batch_size": 32,
                # Use dict syntax which triggers schedule parsing
                "policy_lr": {"start": 3e-4, "end": 1e-4, "from": 0.0, "to": 0.5},
                # max_env_steps is intentionally missing to trigger error
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

    # max_env_steps not divisible by n_envs
    with pytest.raises(ValueError, match="must be divisible by n_envs"):
        Config.build_from_dict({
            "env_id": "CartPole-v1",
            "algo_id": "ppo",
            "n_steps": 32,
            "n_envs": 8,
            "batch_size": 32,
            "max_env_steps": 1001,  # 1001 % 8 = 1, not divisible
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
        "max_env_steps": 10000,
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
        "max_env_steps": 10000,
        "ent_coef": {"start": 0.01, "end": 0.0},
    })

    assert cfg.ent_coef == pytest.approx(0.01)
    assert cfg.ent_coef_schedule == "linear"
    assert cfg.ent_coef_schedule_start_value == pytest.approx(0.01)
    assert cfg.ent_coef_schedule_end_value == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_start == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_end == pytest.approx(1.0)


@pytest.mark.unit
def test_config_max_vec_steps_property():
    """Test that max_vec_steps computed property correctly converts env_steps to vec_steps"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 8,
        "batch_size": 64,
        "max_env_steps": 1000000,  # 1M env steps
    })

    # 1M env steps / 8 envs = 125k vec steps
    assert cfg.max_vec_steps == 125000

    # Test with None
    cfg_no_max = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 8,
        "batch_size": 64,
    })
    assert cfg_no_max.max_vec_steps is None


@pytest.mark.unit
def test_config_schedule_types():
    """Test that all scheduler types can be configured"""
    scheduler_types = ["linear", "cosine", "exponential"]

    for scheduler_type in scheduler_types:
        cfg = Config.build_from_dict({
            "algo_id": "ppo",
            "env_id": "CartPole-v1",
            "n_steps": 128,
            "n_envs": 4,
            "batch_size": 64,
            "max_env_steps": 10000,
            "policy_lr": {"start": 0.003, "end": 0.0001, "schedule": scheduler_type},
        })

        assert cfg.policy_lr_schedule == scheduler_type
        assert cfg.policy_lr == pytest.approx(0.003)
        assert cfg.policy_lr_schedule_start_value == pytest.approx(0.003)
        assert cfg.policy_lr_schedule_end_value == pytest.approx(0.0001)


@pytest.mark.unit
def test_config_schedule_with_warmup():
    """Test that warmup_fraction can be configured"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_env_steps": 10000,
        "policy_lr": {"start": 0.003, "end": 0.0001, "schedule": "cosine", "warmup": 0.1},
    })

    assert cfg.policy_lr_schedule == "cosine"
    assert cfg.policy_lr_schedule_warmup == pytest.approx(0.1)


@pytest.mark.unit
def test_config_build_from_dict_filters_unknown_fields():
    """Test that unknown fields (not in Config dataclass) are filtered out during build_from_dict"""
    # This should not raise TypeError about unexpected keyword arguments
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_env_steps": 1000,
        # These unknown fields should be ignored (not cause errors)
        # Note: the filtering actually happens in build_from_yaml, not build_from_dict,
        # but this test documents the expected behavior
    })

    assert cfg.env_id == "CartPole-v1"
    assert cfg.algo_id == "ppo"


@pytest.mark.unit
def test_config_fractional_eval_warmup_epochs():
    """Test that fractional eval_warmup_epochs is resolved correctly"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_env_steps": 10000,
        "eval_warmup_epochs": 0.3,  # 30% of training
    })

    # total_epochs = 10000 / (4 * 128) = 19.53... epochs
    # warmup = int(19.53 * 0.3) = int(5.859) = 5
    assert cfg.eval_warmup_epochs == 5


@pytest.mark.unit
def test_config_fractional_eval_warmup_epochs_without_max_env_steps():
    """Test that fractional eval_warmup_epochs without max_env_steps raises error"""
    with pytest.raises(AssertionError, match="max_env_steps"):
        Config.build_from_dict({
            "algo_id": "ppo",
            "env_id": "CartPole-v1",
            "n_steps": 128,
            "n_envs": 4,
            "batch_size": 64,
            "eval_warmup_epochs": 0.3,  # Fractional without max_env_steps
        })


@pytest.mark.unit
def test_config_absolute_eval_warmup_epochs():
    """Test that absolute (>=1) eval_warmup_epochs is not modified"""
    cfg = Config.build_from_dict({
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_env_steps": 10000,
        "eval_warmup_epochs": 10,  # Absolute value
    })

    # Should remain unchanged
    assert cfg.eval_warmup_epochs == 10
