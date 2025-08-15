import types
import pytest

from wrappers.env_wrapper_registry import EnvWrapperRegistry


class DummyWrapper:
    def __init__(self, env, factor=1):
        self.env = env
        self.factor = factor


def test_registry_register_and_apply():
    EnvWrapperRegistry.register(DummyWrapper)
    base = types.SimpleNamespace()
    wrapped = EnvWrapperRegistry.apply(base, {"id": "DummyWrapper", "factor": 3})
    assert isinstance(wrapped, DummyWrapper)
    assert wrapped.env is base
    assert wrapped.factor == 3
