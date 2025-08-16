import importlib
import sys
import types

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


def test_sanitize_name():
    assert BaseAgent._sanitize_name("ALE/Pong-v5") == "ALE-Pong-v5"
    assert BaseAgent._sanitize_name("CartPole\\v1") == "CartPole-v1"
    assert BaseAgent._sanitize_name("NoSeparators") == "NoSeparators"


def test_compute_validation_controls_none():
    controls = BaseAgent._compute_validation_controls(None)
    assert controls["limit_val_batches"] == 0
    assert controls["check_val_every_n_epoch"] == 1


def test_compute_validation_controls_positive():
    controls = BaseAgent._compute_validation_controls(5)
    assert controls["limit_val_batches"] == 1.0
    assert controls["check_val_every_n_epoch"] == 5
