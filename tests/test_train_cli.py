import sys


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

    monkeypatch.setattr("agents.build_agent", fake_build_agent)
    monkeypatch.setattr(
        "utils.wandb_workspace.create_or_update_workspace_for_current_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "Bandit-v0:ppo", "--max-steps", "123"])

    train.main()

    assert "config" in captured
    assert captured["config"].max_timesteps == 123
