import importlib
import sys
import threading
import time
import types
from contextlib import contextmanager
from types import SimpleNamespace

import torch

# Provide a lightweight stub for the 'callbacks' module to avoid optional deps (e.g., watchdog)
module = types.ModuleType("callbacks")
class _Dummy:  # minimal callable placeholders used by BaseAgent.train
    def __init__(self, *a, **k):
        pass

module.PrintMetricsCallback = _Dummy
module.VideoLoggerCallback = _Dummy
module.ModelCheckpointCallback = _Dummy
module.HyperparameterScheduler = _Dummy
sys.modules.setdefault("callbacks", module)

BaseAgent = importlib.import_module("agents.base_agent").BaseAgent
base_agent_optimization = importlib.import_module("agents.base_agent_optimization")
base_agent_runtime = importlib.import_module("agents.base_agent_runtime")


def _build_agent(*, max_env_steps=100, total_steps=25, async_metrics=None):
    inst = BaseAgent.__new__(BaseAgent)
    object.__setattr__(inst, "config", SimpleNamespace(max_env_steps=max_env_steps))
    object.__setattr__(inst, "_rollout_collectors", {"train": SimpleNamespace(total_steps=total_steps)})
    object.__setattr__(inst, "_async_eval_lock", threading.Lock())
    object.__setattr__(inst, "_async_eval_metrics", async_metrics or {})
    return inst


def test_calc_training_progress_returns_fraction_of_budget():
    inst = _build_agent(max_env_steps=100, total_steps=25)
    assert inst.calc_training_progress() == 0.25


def test_calc_training_progress_clamps_to_zero_without_budget():
    inst = _build_agent(max_env_steps=None, total_steps=25)
    assert inst.calc_training_progress() == 0.0


def test_set_early_stop_reason_updates_state():
    inst = _build_agent()
    inst.set_early_stop_reason("budget exhausted")
    assert inst._early_stop_reason == "budget exhausted"


def test_get_async_eval_metric_reads_shared_results():
    inst = _build_agent(async_metrics={"val/roll/ep_rew/mean": 123.4})
    assert inst.get_async_eval_metric("val/roll/ep_rew/mean") == 123.4
    assert inst.get_async_eval_metric("missing") is None


def test_train_dataloader_bootstraps_initial_rollout_and_loader(monkeypatch):
    built = {}

    class _Collector:
        def __init__(self):
            self.collect_calls = 0

        def collect(self):
            self.collect_calls += 1
            return SimpleNamespace(observations=torch.zeros(8, 1))

    def fake_build_loader(**kwargs):
        built.update(kwargs)
        return "loader"

    monkeypatch.setattr(
        "utils.dataloaders.build_index_collate_loader_from_collector",
        fake_build_loader,
    )
    monkeypatch.setattr(
        "utils.random.get_global_torch_generator",
        lambda seed: f"generator-{seed}",
    )

    collector = _Collector()
    inst = BaseAgent.__new__(BaseAgent)
    object.__setattr__(inst, "config", SimpleNamespace(seed=7, batch_size=4, n_epochs=3))
    object.__setattr__(inst, "_rollout_collectors", {"train": collector})
    object.__setattr__(inst, "_trajectories", [])
    object.__setattr__(inst, "current_epoch", 0)

    loader = inst.train_dataloader()

    assert loader == "loader"
    assert inst._train_dataloader == "loader"
    assert collector.collect_calls == 1
    assert built["collector"] is collector
    assert built["trajectories_getter"]() is inst._trajectories
    assert built["batch_size"] == 4
    assert built["num_passes"] == 3
    assert built["generator"] == "generator-7"


def test_on_train_epoch_start_stops_before_overshooting_budget():
    events = []

    class _Timings:
        def start(self, *_args, **_kwargs):
            events.append("timings:start")

    class _Collector:
        def __init__(self):
            self.collect_calls = 0

        def get_metrics(self):
            return {"cnt/total_env_steps": 96}

        def collect(self):
            self.collect_calls += 1
            return None

    collector = _Collector()
    inst = BaseAgent.__new__(BaseAgent)
    object.__setattr__(
        inst,
        "config",
        SimpleNamespace(eval_async=False, max_env_steps=100, n_envs=2, n_steps=3),
    )
    object.__setattr__(inst, "_print_metrics_logger", None)
    object.__setattr__(inst, "_rollout_collectors", {"train": collector})
    object.__setattr__(inst, "timings", _Timings())
    object.__setattr__(inst, "trainer", SimpleNamespace(should_stop=False))
    object.__setattr__(inst, "current_epoch", 1)
    object.__setattr__(inst, "_early_stop_reason", "")
    object.__setattr__(inst, "_read_hyperparameters_from_run", lambda: events.append("read"))
    object.__setattr__(inst, "_log_hyperparameters", lambda: events.append("log"))

    inst.on_train_epoch_start()

    assert inst.trainer.should_stop is True
    assert collector.collect_calls == 0
    assert "would exceed" in inst._early_stop_reason
    assert events == ["timings:start", "read", "log"]


def test_validation_hooks_record_async_eval_metrics():
    class _Timings:
        def start(self, *_args, **_kwargs):
            return None

        def throughput_since(self, *_args, **_kwargs):
            return {"cnt/total_vec_steps": 42.0}

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record(self, stage, metrics):
            self.calls.append((stage, metrics))

    class _Collector:
        def get_metrics(self):
            return {"cnt/total_vec_steps": 0}

        def evaluate_episodes(self, *, n_episodes, deterministic):
            assert n_episodes == 3
            assert deterministic is True
            return {"roll/ep_rew/mean": 7.5, "cnt/total_vec_steps": 12}

    inst = BaseAgent.__new__(BaseAgent)
    model = torch.nn.Linear(1, 1)
    object.__setattr__(
        inst,
        "config",
        SimpleNamespace(eval_async=True, eval_episodes=3, eval_deterministic=True),
    )
    object.__setattr__(inst, "_print_metrics_logger", None)
    object.__setattr__(inst, "_rollout_collectors", {"val": _Collector()})
    object.__setattr__(inst, "timings", _Timings())
    object.__setattr__(inst, "metrics_recorder", _Recorder())
    object.__setattr__(inst, "_async_eval_thread", None)
    object.__setattr__(inst, "_async_eval_metrics", {})
    object.__setattr__(inst, "_async_eval_lock", threading.Lock())
    object.__setattr__(inst, "_async_eval_shutdown", threading.Event())
    object.__setattr__(inst, "_async_eval_pending_epoch", None)
    object.__setattr__(inst, "_async_eval_running_epoch", None)
    object.__setattr__(inst, "policy_model", model)
    object.__setattr__(inst, "_eval_models", {"val": torch.nn.Linear(1, 1)})
    object.__setattr__(inst, "trainer", SimpleNamespace(callbacks=[]))
    object.__setattr__(inst, "current_epoch", 5)

    inst.on_validation_epoch_start()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if inst.get_async_eval_metric("eval/model_epoch") == 5:
            break
        time.sleep(0.01)

    assert inst.get_async_eval_metric("eval/model_epoch") == 5
    assert inst.get_async_eval_metric("roll/ep_rew/mean") == 7.5

    inst.on_validation_epoch_end()
    assert inst.metrics_recorder.calls[-1][0] == "val"
    assert inst.metrics_recorder.calls[-1][1]["eval/model_epoch"] == 5


def test_validation_step_refreshes_sync_eval_model_copy():
    class _Timings:
        def throughput_since(self, *_args, **_kwargs):
            return {"cnt/total_vec_steps": 42.0}

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record(self, stage, metrics):
            self.calls.append((stage, metrics))

    class _Collector:
        def __init__(self):
            self.evaluate_calls = 0

        def evaluate_episodes(self, *, n_episodes, deterministic):
            self.evaluate_calls += 1
            assert n_episodes == 2
            assert deterministic is False
            return {"roll/ep_rew/mean": 3.5, "cnt/total_vec_steps": 9}

    inst = BaseAgent.__new__(BaseAgent)
    policy_model = torch.nn.Linear(1, 1)
    eval_model = torch.nn.Linear(1, 1)
    collector = _Collector()
    object.__setattr__(
        inst,
        "config",
        SimpleNamespace(eval_async=False, eval_episodes=2, eval_deterministic=False),
    )
    object.__setattr__(inst, "_print_metrics_logger", None)
    object.__setattr__(inst, "_rollout_collectors", {"val": collector})
    object.__setattr__(inst, "timings", _Timings())
    object.__setattr__(inst, "metrics_recorder", _Recorder())
    object.__setattr__(inst, "policy_model", policy_model)
    object.__setattr__(inst, "_eval_models", {"val": eval_model})
    object.__setattr__(inst, "current_epoch", 4)

    with torch.no_grad():
        for param in policy_model.parameters():
            param.fill_(7.0)
        for param in eval_model.parameters():
            param.zero_()

    inst.validation_step(None, 0)

    for eval_param, policy_param in zip(eval_model.parameters(), policy_model.parameters()):
        assert torch.allclose(eval_param, policy_param)
    assert collector.evaluate_calls == 1
    assert inst.metrics_recorder.calls[-1][0] == "val"
    assert inst.metrics_recorder.calls[-1][1]["cnt/epoch"] == 4


def test_training_step_skips_activation_metrics_when_detailed_metrics_disabled():
    class _Policy:
        def __init__(self):
            self._track_activations = False
            self.activation_stats_calls = 0

        def compute_activation_stats(self):
            self.activation_stats_calls += 1
            return {"opt/activations/backbone/mean": 1.0}

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record(self, stage, metrics):
            self.calls.append((stage, metrics))

    inst = BaseAgent.__new__(BaseAgent)
    object.__setattr__(inst, "_early_stop_epoch", False)
    object.__setattr__(inst, "config", SimpleNamespace(detailed_optimization_metrics=False))
    object.__setattr__(inst, "policy_model", _Policy())
    object.__setattr__(inst, "metrics_recorder", _Recorder())
    object.__setattr__(inst, "_backpropagate_and_step", lambda losses: None)
    object.__setattr__(inst, "losses_for_batch", lambda batch, batch_idx: {"loss": torch.tensor(1.0), "early_stop_epoch": False})

    base_agent_runtime.training_step(inst, None, 0)

    assert inst.policy_model._track_activations is False
    assert inst.policy_model.activation_stats_calls == 0
    assert inst.metrics_recorder.calls == []


def test_backpropagate_and_step_skips_grad_metrics_when_detailed_metrics_disabled():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    grad_metric_calls = {"count": 0}

    def _compute_grad_norms():
        grad_metric_calls["count"] += 1
        return {"opt/grads/norm/all": 1.0}

    model.compute_grad_norms = _compute_grad_norms  # type: ignore[attr-defined]

    class _Recorder:
        def __init__(self):
            self.calls = []

        def record(self, stage, metrics):
            self.calls.append((stage, metrics))

    agent = SimpleNamespace(
        optimizers=lambda: optimizer,
        policy_model=model,
        config=SimpleNamespace(detailed_optimization_metrics=False, max_grad_norm=None),
        metrics_recorder=_Recorder(),
        manual_backward=lambda loss: loss.backward(),
        clip_gradients=lambda *args, **kwargs: None,
    )

    loss = model(torch.ones(1, 1)).sum()
    base_agent_optimization.backpropagate_and_step(agent, loss)

    assert grad_metric_calls["count"] == 0
    assert agent.metrics_recorder.calls == []


def test_learn_creates_run_and_redirects_output(monkeypatch, tmp_path):
    events = []

    @contextmanager
    def fake_stream_output_to_log(path):
        events.append(("stream", path))
        events.append("stream:enter")
        try:
            yield
        finally:
            events.append("stream:exit")

    run_stub = SimpleNamespace(
        run_id="run-123",
        _ensure_path=lambda name: tmp_path / name,
    )

    def fake_create(run_id, config):
        events.append(("create", run_id, config))
        return run_stub

    monkeypatch.setattr("utils.logging.stream_output_to_log", fake_stream_output_to_log)

    import agents.base_agent_runtime as runtime

    monkeypatch.setattr(runtime.wandb, "run", None, raising=False)
    monkeypatch.setattr(runtime.Run, "create", staticmethod(fake_create))

    inst = BaseAgent.__new__(BaseAgent)
    object.__setattr__(inst, "run", None)
    object.__setattr__(inst, "config", SimpleNamespace())
    object.__setattr__(inst, "_learn", lambda: events.append("learn"))

    inst.learn()

    assert inst.run is run_stub
    assert events[0][0] == "create"
    assert events[0][1].startswith("local-")
    assert events[1] == ("stream", tmp_path / "run.log")
    assert events[2:] == ["stream:enter", "learn", "stream:exit"]


def test_learn_runtime_builds_trainer_and_respects_resume_epoch(monkeypatch):
    events = []
    built = {}

    trainer_loggers_mod = types.ModuleType("utils.trainer_loggers")
    callback_builder_mod = types.ModuleType("utils.callback_builder")
    trainer_factory_mod = types.ModuleType("utils.trainer_factory")

    class _TrainerLoggersBuilder:
        def __init__(self, agent):
            events.append("loggers:init")
            self.agent = agent

        def build(self):
            events.append("loggers:build")
            return ["logger"]

    class _CallbackBuilder:
        def __init__(self, agent):
            events.append("callbacks:init")
            self.agent = agent

        def build(self):
            events.append("callbacks:build")
            return ["callback"]

    def fake_build_trainer(*, config, logger, callbacks):
        events.append(("trainer:build", logger, callbacks, config))
        trainer = SimpleNamespace(
            fit=lambda agent: events.append(("trainer:fit", agent)),
            fit_loop=SimpleNamespace(
                epoch_progress=SimpleNamespace(
                    current=SimpleNamespace(completed=0, processed=0)
                )
            ),
        )
        built["trainer"] = trainer
        return trainer

    trainer_loggers_mod.TrainerLoggersBuilder = _TrainerLoggersBuilder
    callback_builder_mod.CallbackBuilder = _CallbackBuilder
    trainer_factory_mod.build_trainer = fake_build_trainer
    monkeypatch.setitem(sys.modules, "utils.trainer_loggers", trainer_loggers_mod)
    monkeypatch.setitem(sys.modules, "utils.callback_builder", callback_builder_mod)
    monkeypatch.setitem(sys.modules, "utils.trainer_factory", trainer_factory_mod)

    inst = BaseAgent.__new__(BaseAgent)
    config = SimpleNamespace()
    object.__setattr__(inst, "config", config)
    object.__setattr__(inst, "run", SimpleNamespace())
    object.__setattr__(inst, "_resume_from_epoch", 3)

    inst._learn()

    trainer = built["trainer"]
    assert trainer.fit_loop.epoch_progress.current.completed == 3
    assert trainer.fit_loop.epoch_progress.current.processed == 3
    assert events == [
        "loggers:init",
        "loggers:build",
        "callbacks:init",
        "callbacks:build",
        ("trainer:build", ["logger"], ["callback"], config),
        ("trainer:fit", inst),
    ]


def test_save_and_load_checkpoint_restore_public_training_state(tmp_path):
    from utils.config import load_config

    cfg = load_config("CartPole-v1", "ppo")
    inst = BaseAgent.__new__(BaseAgent)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    seed_input = torch.ones(1, 2)
    loss = model(seed_input).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    original_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    train_collector = SimpleNamespace(
        _best_episode_reward=7.0,
        total_steps=0,
        total_vec_steps=0,
        get_metrics=lambda: {"cnt/total_env_steps": 128, "cnt/total_vec_steps": 64},
    )
    val_collector = SimpleNamespace(_best_episode_reward=9.0)

    object.__setattr__(inst, "config", cfg)
    object.__setattr__(inst, "policy_model", model)
    object.__setattr__(inst, "_optimizers", optimizer)
    object.__setattr__(inst, "_rollout_collectors", {"train": train_collector, "val": val_collector})
    object.__setattr__(inst, "run", SimpleNamespace(run_id="run-123"))
    object.__setattr__(inst, "current_epoch", 3)

    inst.save_checkpoint(tmp_path)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    optimizer.param_groups[0]["lr"] = 0.5
    train_collector._best_episode_reward = -1.0
    train_collector.total_steps = 0
    train_collector.total_vec_steps = 0
    val_collector._best_episode_reward = -1.0

    inst.load_checkpoint(tmp_path, resume_training=True)

    for key, value in original_state.items():
        assert torch.equal(model.state_dict()[key], value)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert train_collector._best_episode_reward == 7.0
    assert train_collector.total_steps == 128
    assert train_collector.total_vec_steps == 64
    assert val_collector._best_episode_reward == 9.0
