import sys
from types import SimpleNamespace

from utils.train_launcher import DEFAULT_CONFIG_SPEC, resolve_requested_config_spec


def test_resolve_requested_config_spec_uses_shared_default():
    args = SimpleNamespace(config=None, config_id=None)
    assert resolve_requested_config_spec(args) == DEFAULT_CONFIG_SPEC


def test_max_steps_override(monkeypatch):
    import train

    captured = {}

    class DummyTimings:
        def seconds_since(self, *_):
            return 0

    class DummyAgent:
        def __init__(self, config):
            self.config = config
            self.timings = DummyTimings()
            self._fit_elapsed_seconds = 0

        def learn(self):
            pass

    def fake_build_agent(config):
        captured["config"] = config
        return DummyAgent(config)

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("agents.build_agent", fake_build_agent)
    monkeypatch.setattr("utils.train_launcher.build_agent", fake_build_agent)
    monkeypatch.setattr("utils.train_launcher._ensure_wandb_run_initialized", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "utils.wandb_workspace.create_or_update_workspace_for_current_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "Bandit-v0:ppo", "--max-env-steps", "123"])

    train.main()

    assert "config" in captured
    assert captured["config"].max_env_steps == 123
