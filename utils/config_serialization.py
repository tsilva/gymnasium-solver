"""Serialization helpers for Config objects."""

from __future__ import annotations

from dataclasses import asdict


def config_to_dict(config) -> dict:
    data = asdict(config)
    data.pop("_hidden_dims", None)
    data.pop("_activation", None)
    data.pop("_policy_kwargs", None)
    data["algo_id"] = config.algo_id

    for attr_name in dir(config):
        if "_schedule" not in attr_name or attr_name.startswith("_"):
            continue
        value = getattr(config, attr_name, None)
        if value is not None:
            data[attr_name] = value

    return data
