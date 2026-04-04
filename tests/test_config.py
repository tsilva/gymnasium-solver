import pytest

from utils.config import Config


def _config_dict(**overrides):
    base = {
        "algo_id": "ppo",
        "env_id": "CartPole-v1",
        "model_id": "mlp_small",
        "n_steps": 128,
        "n_envs": 4,
        "batch_size": 64,
        "max_env_steps": 10000,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_config_parse_schedules_with_dict_syntax():
    cfg = Config.build_from_dict(
        _config_dict(
            n_steps=32,
            batch_size=16,
            max_env_steps=1000,
            policy_lr={"start": 3e-4, "end": 0.0},
        )
    )
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
            _config_dict(
                n_steps=32,
                batch_size=32,
                max_env_steps=None,
                policy_lr={"start": 3e-4, "end": 1e-4, "from": 0.0, "to": 0.5},
            )
        )

@pytest.mark.unit
def test_config_validate_errors():
    # Invalid learning rate
    with pytest.raises(ValueError):
        Config.build_from_dict(_config_dict(n_steps=32, batch_size=32, policy_lr=0.0))

    # Invalid gamma
    with pytest.raises(ValueError):
        Config.build_from_dict(_config_dict(n_steps=32, batch_size=32, gamma=1.5))

    # max_env_steps is rounded to match rollout divisibility in current runtime behavior
    cfg = Config.build_from_dict(_config_dict(n_envs=8, batch_size=32, max_env_steps=1001))
    assert cfg.max_env_steps == 1000


@pytest.mark.unit
def test_config_schedule_dict_syntax():
    """Test new dict-based schedule syntax"""
    cfg = Config.build_from_dict(
        _config_dict(
            ent_coef={"start": 0.02, "end": 0.001},
            policy_lr={"start": 0.003, "end": 0.0001, "from": 0.0, "to": 0.8},
            clip_range={"start": 0.2, "end": 0.05, "schedule": "linear"},
        )
    )

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
    cfg = Config.build_from_dict(_config_dict(ent_coef={"start": 0.01, "end": 0.0}))

    assert cfg.ent_coef == pytest.approx(0.01)
    assert cfg.ent_coef_schedule == "linear"
    assert cfg.ent_coef_schedule_start_value == pytest.approx(0.01)
    assert cfg.ent_coef_schedule_end_value == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_start == pytest.approx(0.0)
    assert cfg.ent_coef_schedule_end == pytest.approx(1.0)


@pytest.mark.unit
def test_config_max_vec_steps_property():
    """Test that max_vec_steps computed property correctly converts env_steps to vec_steps"""
    cfg = Config.build_from_dict(_config_dict(n_envs=8, max_env_steps=1000000))

    # 1M env steps / 8 envs = 125k vec steps
    assert cfg.max_vec_steps == 125000

    # Test with None
    cfg_no_max = Config.build_from_dict(_config_dict(n_envs=8, max_env_steps=None))
    assert cfg_no_max.max_vec_steps is None


@pytest.mark.unit
def test_config_schedule_types():
    """Test that all scheduler types can be configured"""
    scheduler_types = ["linear", "cosine", "exponential"]

    for scheduler_type in scheduler_types:
        cfg = Config.build_from_dict(
            _config_dict(policy_lr={"start": 0.003, "end": 0.0001, "schedule": scheduler_type})
        )

        assert cfg.policy_lr_schedule == scheduler_type
        assert cfg.policy_lr == pytest.approx(0.003)
        assert cfg.policy_lr_schedule_start_value == pytest.approx(0.003)
        assert cfg.policy_lr_schedule_end_value == pytest.approx(0.0001)


@pytest.mark.unit
def test_config_schedule_with_warmup():
    """Test that warmup_fraction can be configured"""
    cfg = Config.build_from_dict(
        _config_dict(policy_lr={"start": 0.003, "end": 0.0001, "schedule": "cosine", "warmup": 0.1})
    )

    assert cfg.policy_lr_schedule == "cosine"
    assert cfg.policy_lr_schedule_warmup == pytest.approx(0.1)


@pytest.mark.unit
def test_config_build_from_dict_filters_unknown_fields():
    """Test that unknown fields (not in Config dataclass) are filtered out during build_from_dict"""
    # This should not raise TypeError about unexpected keyword arguments
    cfg = Config.build_from_dict(_config_dict(max_env_steps=1000, unknown_field="ignored"))

    assert cfg.env_id == "CartPole-v1"
    assert cfg.algo_id == "ppo"


@pytest.mark.unit
def test_config_fractional_eval_warmup_epochs():
    """Test that fractional eval_warmup_epochs is resolved correctly"""
    cfg = Config.build_from_dict(_config_dict(eval_warmup_epochs=0.3))

    # total_epochs = 10000 / (4 * 128) = 19.53... epochs
    # warmup = int(19.53 * 0.3) = int(5.859) = 5
    assert cfg.eval_warmup_epochs == 5


@pytest.mark.unit
def test_config_fractional_eval_warmup_epochs_without_max_env_steps():
    """Test that fractional eval_warmup_epochs without max_env_steps raises error"""
    with pytest.raises(AssertionError, match="max_env_steps"):
        Config.build_from_dict(_config_dict(max_env_steps=None, eval_warmup_epochs=0.3))


@pytest.mark.unit
def test_config_absolute_eval_warmup_epochs():
    """Test that absolute (>=1) eval_warmup_epochs is not modified"""
    cfg = Config.build_from_dict(_config_dict(eval_warmup_epochs=10))

    # Should remain unchanged
    assert cfg.eval_warmup_epochs == 10
