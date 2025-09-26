"""Utilities for working with environment specifications."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class EnvSpec(Mapping[str, Any]):
    """Immutable view over merged environment spec data."""

    _data: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", dict(self._data))

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Convenience accessors -------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @property
    def id(self) -> Optional[str]:
        value = self._data.get("id")
        return str(value) if value is not None else None

    def get_max_episode_steps(self) -> Optional[int]:
        value = self._data.get("max_episode_steps")
        if not isinstance(value, (int, float)):  return None
        if value <= 0: return None
        return int(value)

    def get_render_fps(self) -> Optional[int]:
        value = self._data.get("render_fps")
        if not isinstance(value, (int, float)): return None
        if value <= 0: return None
        return int(value)

    def get_action_labels(self) -> Dict[str, Any]:
        action_space = self._section("action_space")
        labels = action_space.get("labels", {})
        return dict(labels) if isinstance(labels, Mapping) else {}

    def get_reward_range(self) -> Optional[Sequence[float]]:
        return self._validated_range("rewards", "range", "Reward range")

    def get_return_range(self) -> Optional[Sequence[float]]:
        return self._validated_range("returns", "range", "Return range")

    def get_return_threshold(self) -> Any:
        returns = self._section("returns")
        return returns.get("threshold_solved")

    # -- Internal helpers -------------------------------------------------
    def _section(self, key: str) -> Dict[str, Any]:
        section = self._data.get(key, {})
        return dict(section) if isinstance(section, Mapping) else {}

    def _validated_range(self, section: str, field: str, label: str) -> Optional[Sequence[float]]:
        section_data = self._section(section)
        rng = section_data.get(field)
        if rng is None: return None
        if not isinstance(rng, (list, tuple)): raise ValueError(f"{label} must be a list or tuple, got {type(rng)}")
        if len(rng) != 2: raise ValueError(f"{label} must be a 2-element list or tuple, got {len(rng)}")
        return [rng[0], rng[1]]
