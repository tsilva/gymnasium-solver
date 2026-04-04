"""BaseAgent helpers for environment and rollout collector setup."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def build_stage_env(agent: "BaseAgent", stage: str, **kwargs) -> None:
    """Build and attach the vectorized environment for one lifecycle stage."""
    from utils.environment import build_env_from_config, get_env_type

    agent._envs = agent._envs if hasattr(agent, "_envs") else {}

    if agent.config.eval_async and stage == "val" and "n_envs" not in kwargs:
        eval_n_envs = max(4, min(8, agent.config.n_envs // 4))
        kwargs["n_envs"] = eval_n_envs

    reuse_alepy_vectorization = (
        get_env_type(agent.config.env_id) == "alepy"
        and getattr(agent.config, "obs_type", None) == "rgb"
        and agent.config.vectorization_mode in ("auto", "alepy")
        and "vectorization_mode" not in kwargs
    )

    default_kwargs = {
        "train": {
            "seed": agent.config.seed_train,
        },
        "val": {
            "seed": agent.config.seed_val,
            "vectorization_mode": "sync",
            "render_mode": "rgb_array",
            "record_video": False,
            "record_video_kwargs": {
                "video_length": 100,
            },
        },
        "test": {
            "seed": agent.config.seed_test,
            "vectorization_mode": "sync",
            "render_mode": "rgb_array",
            "record_video": False,
            "record_video_kwargs": {
                "video_length": None,
            },
        },
    }

    if reuse_alepy_vectorization:
        default_kwargs["val"]["vectorization_mode"] = agent.config.vectorization_mode
        default_kwargs["test"]["vectorization_mode"] = agent.config.vectorization_mode

    if get_env_type(agent.config.env_id) == "stable_retro" and stage in ("val", "test"):
        kwargs.setdefault("n_envs", 1)
        kwargs["vectorization_mode"] = "async"
        kwargs["record_video"] = False

    agent._envs[stage] = build_env_from_config(
        agent.config,
        **{
            **default_kwargs[stage],
            **kwargs,
        },
    )


def build_stage_rollout_collector(agent: "BaseAgent", stage: str) -> None:
    """Build and attach the rollout collector for one lifecycle stage."""
    from utils.rollout_collector import RolloutCollector

    agent._rollout_collectors = (
        agent._rollout_collectors if hasattr(agent, "_rollout_collectors") else {}
    )

    if stage in ("val", "test"):
        policy_copy = copy.deepcopy(agent.policy_model)
        if not hasattr(agent, "_eval_models"):
            agent._eval_models = {}
        agent._eval_models[stage] = policy_copy
        model_for_collector = policy_copy
    else:
        model_for_collector = agent.policy_model

    agent._rollout_collectors[stage] = RolloutCollector(
        agent.get_env(stage),
        model_for_collector,
        **agent.config.get_rollout_collector_kwargs(),
    )
