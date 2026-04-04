import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch


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
    """Install a lightweight utils.environment module exposing current builder APIs."""
    env_mod = ModuleType("utils.environment")

    def build_env(*args, **kwargs):  # noqa: ARG001
        n_envs = kwargs.get("n_envs", 1)
        return DummyVecEnvForPPO(num_envs=n_envs)

    def build_env_from_config(_config, **kwargs):
        return build_env(**kwargs)

    env_mod.build_env = build_env
    env_mod.build_env_from_config = build_env_from_config
    env_mod.get_env_type = lambda _env_id: None
    # Ensure utils package exists in sys.modules for submodule registration
    if "utils" not in sys.modules:
        sys.modules["utils"] = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils.environment", env_mod)


def _install_training_support_stubs(monkeypatch):
    loggers_mod = ModuleType("utils.trainer_loggers")

    class _TrainerLoggersBuilder:
        def __init__(self, agent):  # noqa: ANN001
            self.agent = agent

        def build(self):
            return []

    loggers_mod.TrainerLoggersBuilder = _TrainerLoggersBuilder

    callbacks_mod = ModuleType("utils.callback_builder")

    class _CallbackBuilder:
        def __init__(self, agent):  # noqa: ANN001
            self.agent = agent

        def build(self):
            return []

    callbacks_mod.CallbackBuilder = _CallbackBuilder

    logging_mod = ModuleType("utils.logging")

    @contextmanager
    def stream_output_to_log(_path):
        yield

    logging_mod.stream_output_to_log = stream_output_to_log

    if "utils" not in sys.modules:
        sys.modules["utils"] = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils.trainer_loggers", loggers_mod)
    monkeypatch.setitem(sys.modules, "utils.callback_builder", callbacks_mod)
    monkeypatch.setitem(sys.modules, "utils.logging", logging_mod)


def _install_trainer_factory_stub(monkeypatch):
    """Install a minimal utils.trainer_factory with a no-PL Trainer stub."""
    tf_mod = ModuleType("utils.trainer_factory")

    class _Trainer:
        def __init__(self, *, max_epochs=1, **_):
            self.max_epochs = 1 if max_epochs is None else max_epochs
            self.should_stop = False
            self.callbacks = []
            self.fit_loop = SimpleNamespace(
                epoch_progress=SimpleNamespace(
                    current=SimpleNamespace(completed=0, processed=0)
                )
            )

        def fit(self, lightning_module):
            # Minimal emulation of PL fit loop used in BaseAgent
            lightning_module.trainer = self
            if not hasattr(lightning_module, "clip_gradients"):
                lightning_module.clip_gradients = lambda optimizer, gradient_clip_val, gradient_clip_algorithm: torch.nn.utils.clip_grad_norm_(  # noqa: ARG005
                    lightning_module.policy_model.parameters(),
                    gradient_clip_val,
                )
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

                metrics = lightning_module.get_rollout_collector("train").get_metrics()
                max_env_steps = getattr(lightning_module.config, "max_env_steps", None)
                if max_env_steps is not None and metrics.get("cnt/total_env_steps", 0) >= max_env_steps:
                    break
                if self.should_stop:
                    break

            if hasattr(lightning_module, "on_fit_end"):
                lightning_module.on_fit_end()

    def build_trainer(*, config, logger, callbacks):  # noqa: ARG001
        trainer = _Trainer(max_epochs=config.max_epochs)
        trainer.callbacks = callbacks
        return trainer

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
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.single_action_space = gym.spaces.Discrete(self._action_dim)
        self.action_space = self.single_action_space

    def reset(self, seed=None):
        del seed
        self._step.fill(0)
        self._obs.fill(0.0)
        return self._obs.copy(), {}

    def step(self, actions):  # actions ignored; deterministic dynamics
        self._step += 1
        self._obs = self._obs + 0.1  # simple drift
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = self._step % self._episode_len == 0
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = {
            "episode": {
                "r": np.where(terminated, float(self._episode_len), 0.0).astype(np.float32),
                "l": np.where(terminated, int(self._episode_len), 0).astype(np.int32),
            },
            "_episode": terminated.copy(),
        }
        if np.any(terminated):
            self._step[terminated] = 0
            self._obs[terminated] = 0.0
        return self._obs.copy(), rewards, terminated, truncated, infos

    # Minimal video recorder context manager used at the end of training
    def recorder(self, *_args, **_kwargs):
        class _Rec:
            def __enter__(_self):
                return self

            def __exit__(_self, exc_type, exc, tb):  # noqa: ARG002
                return False

        return _Rec()

    def close(self):
        return None


@pytest.mark.unit
def test_full_ppo_train_tiny_loop_progress(monkeypatch, tmp_path):
    _install_environment_stub(monkeypatch)
    _install_training_support_stubs(monkeypatch)
    _install_trainer_factory_stub(monkeypatch)

    # Build the default config (as train.py would) and force a tiny run
    from utils.config import load_config

    cfg = load_config("CartPole-v1", "ppo")

    from pathlib import Path
    import agents.base_agent as base_agent_mod

    run_stub = SimpleNamespace(
        run_id="test-run",
        checkpoints_dir=Path(tmp_path),
        _ensure_path=lambda name: Path(tmp_path) / name,
        load_config=lambda: cfg,
    )
    monkeypatch.setattr(base_agent_mod.Run, "create", staticmethod(lambda run_id, config: run_stub), raising=True)

    # Keep it minimal and fast
    cfg.n_envs = 1
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.n_epochs = 1
    cfg.max_epochs = 1
    cfg.max_env_steps = 8  # single rollout worth of env steps
    cfg.enable_wandb = False
    # Disable eval to avoid validation hooks entirely
    cfg.eval_async = False
    cfg.eval_episodes = 1
    # Simpler runtime
    cfg.accelerator = "cpu"
    cfg.devices = None

    # Create agent and run training end-to-end
    from agents import build_agent

    agent = build_agent(cfg)
    agent.learn()

    # Verify progress was recorded and loop didn't crash
    m = agent.get_rollout_collector("train").get_metrics()
    assert m["cnt/total_env_steps"] >= cfg.n_steps * cfg.n_envs
    assert m["cnt/total_rollouts"] >= 1
