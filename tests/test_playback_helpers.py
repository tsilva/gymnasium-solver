from types import SimpleNamespace

import pytest

from utils.playback import (
    get_action_label_list_from_env,
    get_action_labels_from_env,
    resolve_playback_seed,
)


class FakeEnv:
    def __init__(self, labels):
        self._labels = labels

    def get_action_labels(self):
        return self._labels


@pytest.mark.unit
def test_resolve_playback_seed_defaults_to_test_seed():
    config = SimpleNamespace(seed_train=1, seed_val=2, seed_test=3)
    assert resolve_playback_seed(config, None) == 3


@pytest.mark.unit
def test_resolve_playback_seed_accepts_named_stage_and_numeric_values():
    config = SimpleNamespace(seed_train=11, seed_val=22, seed_test=33)
    assert resolve_playback_seed(config, "train") == 11
    assert resolve_playback_seed(config, "val") == 22
    assert resolve_playback_seed(config, "7") == 7
    assert resolve_playback_seed(config, 9) == 9


@pytest.mark.unit
def test_get_action_labels_from_env_normalizes_dict_keys():
    env = FakeEnv({"0": "NOOP", 1: "FIRE"})
    assert get_action_labels_from_env(env) == {0: "NOOP", 1: "FIRE"}


@pytest.mark.unit
def test_get_action_labels_from_env_handles_sequences():
    env = FakeEnv(["LEFT", "RIGHT"])
    assert get_action_labels_from_env(env) == {0: "LEFT", 1: "RIGHT"}


@pytest.mark.unit
def test_get_action_label_list_from_env_fills_missing_indices():
    env = FakeEnv({0: "NOOP", 2: "FIRE"})
    assert get_action_label_list_from_env(env) == ["NOOP", "1", "FIRE"]


@pytest.mark.unit
def test_get_action_labels_from_env_rejects_non_integer_keys():
    env = FakeEnv({"left": "LEFT"})
    assert get_action_labels_from_env(env) is None
