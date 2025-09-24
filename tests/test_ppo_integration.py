import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _install_callbacks_stub(monkeypatch):
    mod = ModuleType("callbacks")

    class _Dummy:
        def __init__(self, *a, **k):
            pass

    # Minimal placeholders used by BaseAgent._build_callbacks
    mod.PrintMetricsCallback = _Dummy
    mod.VideoLoggerCallback = _Dummy
    mod.ModelCheckpointCallback = _Dummy
    mod.HyperparameterScheduler = _Dummy
    monkeypatch.setitem(sys.modules, "callbacks", mod)


def _install_wrappers_stub(monkeypatch):
    pkg = ModuleType("wrappers")
    sub = ModuleType("wrappers.env_wrapper_registry")

    class _Registry:
        @staticmethod
        def apply(env, wrapper_name):  # noqa: ARG002 - no-op
            return env

    sub.EnvWrapperRegistry = _Registry
    # Register both the package and the submodule so 'from wrappers.env_wrapper_registry import EnvWrapperRegistry' works
    monkeypatch.setitem(sys.modules, "wrappers", pkg)
    monkeypatch.setitem(sys.modules, "wrappers.env_wrapper_registry", sub)


def _install_environment_stub(monkeypatch):
    """Install a lightweight utils.environment module exposing build_env only."""
    env_mod = ModuleType("utils.environment")

    def build_env(*args, **kwargs):  # noqa: ARG001
        n_envs = kwargs.get("n_envs", 1)
        return DummyVecEnvForPPO(num_envs=n_envs)

    env_mod.build_env = build_env
    # Ensure utils package exists in sys.modules for submodule registration
    if "utils" not in sys.modules:
        sys.modules["utils"] = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils.environment", env_mod)


def _install_trainer_factory_stub(monkeypatch):
    """Install a minimal utils.trainer_factory with a no-PL Trainer stub."""
    tf_mod = ModuleType("utils.trainer_factory")

    class _Trainer:
        def __init__(self, *, max_epochs=1, **_):
            self.max_epochs = 1 if max_epochs is None else max_epochs

        def fit(self, lightning_module):
            # Minimal emulation of PL fit loop used in BaseAgent
            if hasattr(lightning_module, "on_fit_start"):
                lightning_module.on_fit_start()

            epochs = int(self.max_epochs) if self.max_epochs is not None else 1
            for _ep in range(epochs):
                # Emulate PL-maintained epoch counter used by callbacks
                try:
                    setattr(lightning_module, "current_epoch", _ep)
                except Exception:
                    pass
                if hasattr(lightning_module, "on_train_epoch_start"):
                    lightning_module.on_train_epoch_start()

                dl = lightning_module.train_dataloader()
                for batch_idx, batch in enumerate(dl):
                    lightning_module.training_step(batch, batch_idx)

                if hasattr(lightning_module, "on_train_epoch_end"):
                    lightning_module.on_train_epoch_end()

                # Early stop if enough timesteps gathered (mirrors BaseAgent behavior)
                try:
                    metrics = lightning_module.train_collector.get_metrics()
                    max_timesteps = getattr(lightning_module.config, "max_timesteps", None)
                    if max_timesteps is not None and metrics.get("cnt/total_timesteps", 0) >= max_timesteps:
                        break
                except Exception:
                    pass

            if hasattr(lightning_module, "on_fit_end"):
                lightning_module.on_fit_end()

    def build_trainer(*, logger, callbacks, validation_controls, max_epochs, accelerator="cpu", devices=None):  # noqa: ARG001
        return _Trainer(max_epochs=max_epochs)

    tf_mod.build_trainer = build_trainer
    if "utils" not in sys.modules:
        sys.modules["utils"] = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils.trainer_factory", tf_mod)


class DummyVecEnvForPPO:
    """Tiny deterministic vector env with CartPole-like signatures.

    Meets the subset of VecEnv API used by our RolloutCollector and BaseAgent.
    """

    def __init__(self, num_envs=1, obs_dim=4, action_dim=2, episode_len=10):
        self.num_envs = int(num_envs)
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._episode_len = int(episode_len)
        self._step = np.zeros(self.num_envs, dtype=np.int64)
        self._obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)

    def reset(self):
        self._step.fill(0)
        self._obs.fill(0.0)
        return self._obs.copy()

    def step(self, actions):  # actions ignored; deterministic dynamics
        self._step += 1
        self._obs = self._obs + 0.1  # simple drift
        rewards = np.ones(self.num_envs, dtype=np.float32)
        dones = (self._step % self._episode_len == 0)
        infos = [
            ({"episode": {"r": float(self._episode_len), "l": int(self._episode_len)}} if d else {})
            for d in dones
        ]
        return self._obs.copy(), rewards, dones, infos

    # Minimal video recorder context manager used at the end of training
    def recorder(self, *_args, **_kwargs):
        class _Rec:
            def __enter__(_self):
                return self

            def __exit__(_self, exc_type, exc, tb):  # noqa: ARG002
                return False

        return _Rec()


@pytest.mark.unit
def test_full_ppo_train_tiny_loop_progress(monkeypatch):
    # Stub callbacks and wandb logger to avoid external deps/side effects
    _install_callbacks_stub(monkeypatch)
    _install_wrappers_stub(monkeypatch)
    _install_environment_stub(monkeypatch)
    _install_trainer_factory_stub(monkeypatch)

    # Stub wandb logger creation at the BaseAgent level
    import agents.base_agent as base_agent_mod

    def _fake_create_wandb_logger(self):  # noqa: ANN001
        return SimpleNamespace(experiment=SimpleNamespace(id="test-run", define_metric=lambda *a, **k: None))

    monkeypatch.setattr(base_agent_mod.BaseAgent, "_create_wandb_logger", _fake_create_wandb_logger, raising=True)

    # Build the default config (as train.py would) and force a tiny run
    from utils.config import load_config

    cfg = load_config("CartPole-v1_ppo")
    # Keep it minimal and fast
    cfg.n_envs = 1
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.n_epochs = 1
    cfg.max_epochs = 1
    cfg.max_timesteps = 8  # single rollout worth of timesteps
    # Disable eval to avoid validation hooks entirely
    cfg.eval_freq_epochs = None
    cfg.eval_episodes = None
    cfg.eval_recording_freq_epochs = None
    # Simpler runtime
    cfg.accelerator = "cpu"
    cfg.devices = None

    # Create agent and run training end-to-end
    from agents import build_agent

    agent = build_agent(cfg)
    agent.fit()

    # Verify progress was recorded and loop didn't crash
    m = agent.train_collector.get_metrics()
    assert m["cnt/total_timesteps"] >= cfg.n_steps * cfg.n_envs
    assert m["cnt/total_rollouts"] >= 1
    assert m["roll/fps"] > 0.0
