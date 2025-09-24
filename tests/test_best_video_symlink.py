import importlib.util
import sys
import types
from pathlib import Path
from pathlib import Path as _P
from typing import Any

import pytest

# Load the real ModelCheckpointCallback directly from file to avoid the
# test stub that injects a dummy module in sys.modules.
try:
    import pytorch_lightning as _pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency missing
    _pl = types.ModuleType("pytorch_lightning")
    sys.modules["pytorch_lightning"] = _pl

if not hasattr(_pl, "Callback"):
    class _StubCallback:  # pragma: no cover - minimal base for callback subclassing
        def on_validation_epoch_end(self, *_args, **_kwargs):
            return None

        def on_fit_end(self, *_args, **_kwargs):
            return None

    _pl.Callback = _StubCallback  # type: ignore[attr-defined]

_mod_path = _P(__file__).resolve().parents[1] / "trainer_callbacks" / "model_checkpoint.py"
_spec = importlib.util.spec_from_file_location("real_callbacks_model_checkpoint", str(_mod_path))
assert _spec and _spec.loader
_real_mod = importlib.util.module_from_spec(_spec)
sys.modules["real_callbacks_model_checkpoint"] = _real_mod
_spec.loader.exec_module(_real_mod)
ModelCheckpointCallback = _real_mod.ModelCheckpointCallback


class _Agent:
    def __init__(self, tmp: Path):
        # minimal attributes used by callback
        self.current_epoch = 1
        self.global_step = 10
        self.total_timesteps = 10
        self.best_eval_reward = float('-inf')
        # fake config and run_manager
        class _Cfg:
            env_id = "CartPole-v1"
            reward_threshold = None
            early_stop_on_eval_threshold = False
        self.config = _Cfg()
        class _RM:
            def __init__(self, root: Path):
                self.root = root
            def get_checkpoint_dir(self) -> Path:
                p = self.root / "checkpoints"
                p.mkdir(parents=True, exist_ok=True)
                return p
        self.run_manager = _RM(tmp)
        # minimal API expected by checkpoint save
        class _Pol:
            def state_dict(self):
                return {}
        self.policy_model = _Pol()
        self.optimizers = lambda: type("O", (), {"state_dict": lambda self: {}})()
        # minimal validation env API
        class _VEnv:
            def get_return_threshold(self):
                return None
        self.validation_env = _VEnv()


class _Trainer:
    def __init__(self):
        self.logged_metrics = {"val/ep_rew/mean": 1.0}
        self.global_step = 10
        self.should_stop = False


@pytest.mark.unit
def test_best_mp4_symlink_points_to_epoch_video(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    # create an epoch video that should be marked as best
    epoch_video = ckpt_dir / "epoch=01.mp4"
    epoch_video.write_bytes(b"fake mp4 bytes")

    cb = ModelCheckpointCallback(checkpoint_dir=str(ckpt_dir), metric="val/ep_rew/mean")
    trainer = _Trainer()
    agent = _Agent(tmp_path)

    # invoke validation end; since current metric > -inf, it will be best
    cb.on_validation_epoch_end(trainer, agent)
    cb.on_fit_end(trainer, agent)

    best_link = ckpt_dir / "best.mp4"
    assert best_link.exists(), "best.mp4 link not created"
    # Resolve the link if it's a symlink, else it might be a copied file
    if best_link.is_symlink():
        target = (best_link.parent / best_link.readlink()).resolve()
        assert target == epoch_video.resolve()
    else:
        # copy fallback
        assert best_link.read_bytes() == epoch_video.read_bytes()


class _RecorderStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, float]]] = []

    def record(self, namespace: str, metrics: dict[str, float]) -> None:
        self.calls.append((namespace, dict(metrics)))


class _TimingsStub:
    def __init__(self, elapsed: float) -> None:
        self.elapsed = elapsed

    def seconds_since(self, marker: str) -> float:
        if marker != "on_fit_start":
            raise KeyError(marker)
        return self.elapsed


class _LoggingAgent:
    def __init__(self, tmp: Path, elapsed: float) -> None:
        self.current_epoch = 3
        self.policy_model = type("_Policy", (), {"state_dict": lambda self: {}})()
        self.metrics_recorder = _RecorderStub()
        self.timings = _TimingsStub(elapsed)
        self._log_calls: list[tuple[dict[str, float], dict[str, Any]]] = []

        # lightning callback expects checkpoint directory to be pre-created via cb init
        tmp.mkdir(parents=True, exist_ok=True)

    def log_dict(self, data, **kwargs):  # type: ignore[override]
        self._log_calls.append((dict(data), dict(kwargs)))


@pytest.mark.unit
def test_checkpoint_logs_timing_metrics(tmp_path: Path):
    ckpt_dir = tmp_path / "with_timing" / "checkpoints"
    cb = ModelCheckpointCallback(checkpoint_dir=str(ckpt_dir), metric="val/ep_rew/mean")
    trainer = _Trainer()
    trainer.logged_metrics["val/ep_rew/mean"] = 2.5

    elapsed = 12.5
    agent = _LoggingAgent(ckpt_dir, elapsed=elapsed)

    cb.on_validation_epoch_end(trainer, agent)

    # Confirm checkpoint timing metrics were logged via Lightning
    logged_dicts = [payload for payload, _ in agent._log_calls]
    matching = [d for d in logged_dicts if "val/checkpoint/save_duration_s" in d]
    assert matching, "checkpoint timing metrics were not logged"
    entry = matching[-1]
    assert entry["val/checkpoint/time_elapsed_s"] == pytest.approx(elapsed)
    assert entry["val/checkpoint/save_duration_s"] >= 0.0

    # Ensure metrics recorder received the checkpoint metrics namespace
    assert any(
        ns == "val" and "checkpoint/save_duration_s" in metrics
        for ns, metrics in agent.metrics_recorder.calls
    )
