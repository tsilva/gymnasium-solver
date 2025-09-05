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

def test_compute_validation_controls_zero_disables():
    controls = BaseAgent._compute_validation_controls(0)
    assert controls["limit_val_batches"] == 0
    assert controls["check_val_every_n_epoch"] == 1


def test_should_run_eval_no_warmup():
    class Cfg:
        eval_freq_epochs = 5
        eval_warmup_epochs = 0

    # Create a dummy instance with config
    inst = BaseAgent.__new__(BaseAgent)
    inst.config = Cfg()

    # Epochs are 0-based inside the module; E = epoch+1
    # With no warmup: evaluate at E==1 and multiples of 5
    assert inst._should_run_eval(0) is True   # E=1
    assert inst._should_run_eval(1) is False  # E=2
    assert inst._should_run_eval(3) is False  # E=4
    assert inst._should_run_eval(4) is True   # E=5
    assert inst._should_run_eval(9) is True   # E=10


def test_should_run_eval_with_warmup():
    class Cfg:
        eval_freq_epochs = 2
        eval_warmup_epochs = 3

    inst = BaseAgent.__new__(BaseAgent)
    inst.config = Cfg()

    # Before or at warmup (E <= 3): skip
    assert inst._should_run_eval(0) is False  # E=1
    assert inst._should_run_eval(1) is False  # E=2
    assert inst._should_run_eval(2) is False  # E=3 (boundary is skipped)
    # After warmup: align to cadence grid (multiples of 2)
    assert inst._should_run_eval(3) is True   # E=4
    assert inst._should_run_eval(4) is False  # E=5
    assert inst._should_run_eval(5) is True   # E=6


def test_should_run_eval_zero_freq_never_runs():
    class Cfg:
        eval_freq_epochs = 0
        eval_warmup_epochs = 0

    inst = BaseAgent.__new__(BaseAgent)
    inst.config = Cfg()

    for e in range(0, 10):
        assert inst._should_run_eval(e) is False
