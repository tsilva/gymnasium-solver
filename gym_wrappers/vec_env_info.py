from typing import Any
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

class VecEnvInfoWrapper(VecEnvWrapper):
    """
    VecEnv helper that proxies unknown method calls to the first underlying env
    (index 0) via VecEnv.env_method. Works for SubprocVecEnv and DummyVecEnv.

    Known methods/attributes defined on this wrapper (or parents) behave
    normally. Only *missing* attributes are proxied.
    """

    # ---- VecEnv overrides -------------------------------------------------
    def reset(self) -> Any:
        return self.venv.reset()

    def step_wait(self) -> Any:
        return self.venv.step_wait()

    # ---- Internal helper --------------------------------------------------
    def _call_env(self, method: str, *args, **kwargs) -> Any:
        result = self.venv.env_method(method, *args, indices=[0], **kwargs)
        return result[0]

    # ---- Dynamic proxying -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        assert not name.startswith("_"), f"Cannot proxy private/dunder name: {name}"
        def _proxy_method(*args, **kwargs): return self._call_env(name, *args, **kwargs)
        setattr(self, name, _proxy_method)
        return _proxy_method