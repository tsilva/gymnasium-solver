from pathlib import Path

import pytest

import importlib.util
import sys

from pathlib import Path as _P

# Load the real ModelCheckpointCallback directly from file to avoid the
# test stub that injects a dummy 'callbacks' module in sys.modules.
_mod_path = _P(__file__).resolve().parents[1] / "callbacks" / "model_checkpoint.py"
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
            def get_reward_threshold(self):
                return None
        self.validation_env = _VEnv()


class _Trainer:
    def __init__(self):
        self.logged_metrics = {"val/ep_rew_mean": 1.0}
        self.global_step = 10
        self.should_stop = False


@pytest.mark.unit
def test_best_mp4_symlink_points_to_epoch_video(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    # create an epoch video that should be marked as best
    epoch_video = ckpt_dir / "epoch=01.mp4"
    epoch_video.write_bytes(b"fake mp4 bytes")

    cb = ModelCheckpointCallback(checkpoint_dir=str(ckpt_dir))
    trainer = _Trainer()
    agent = _Agent(tmp_path)

    # invoke validation end; since current metric > -inf, it will be best
    cb.on_validation_epoch_end(trainer, agent)

    best_link = ckpt_dir / "best.mp4"
    assert best_link.exists(), "best.mp4 link not created"
    # Resolve the link if it's a symlink, else it might be a copied file
    if best_link.is_symlink():
        target = (best_link.parent / best_link.readlink()).resolve()
        assert target == epoch_video.resolve()
    else:
        # copy fallback
        assert best_link.read_bytes() == epoch_video.read_bytes()
