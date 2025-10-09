from dataclasses import dataclass, replace

from agents.hyperparameter_mixin import HyperparameterMixin


@dataclass
class DummyConfig:
    policy_lr: float
    ent_coef: float
    clip_range: float


class DummyRun:
    def __init__(self, config: DummyConfig):
        self._config = config

    def load_config(self) -> DummyConfig:
        return replace(self._config)


class DummyAgent(HyperparameterMixin):
    def __init__(self, config: DummyConfig, run: DummyRun):
        self.config = config
        self.run = run
        self.policy_lr = config.policy_lr
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range


def test_read_hyperparameters_skips_scheduled_parameters():
    run_config = DummyConfig(policy_lr=0.0055, ent_coef=0.04, clip_range=0.3)
    agent_config = DummyConfig(policy_lr=0.0055, ent_coef=0.04, clip_range=0.3)

    # Mimic schedule attributes populated by config loader
    agent_config.policy_lr_schedule = "linear"
    agent_config.ent_coef_schedule = "linear"
    agent_config.clip_range_schedule = "linear"

    agent = DummyAgent(agent_config, DummyRun(run_config))

    # Simulate scheduler updates taking effect during training
    agent.policy_lr = agent_config.policy_lr = 0.001
    agent.ent_coef = agent_config.ent_coef = 0.02
    agent.clip_range = agent_config.clip_range = 0.2

    # Run config still has original values; method should respect schedules and keep updates
    agent._read_hyperparameters_from_run()

    assert agent.policy_lr == 0.001
    assert agent.ent_coef == 0.02
    assert agent.clip_range == 0.2
