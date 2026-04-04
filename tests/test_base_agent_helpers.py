import importlib
import sys
import threading
import time
import types
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
