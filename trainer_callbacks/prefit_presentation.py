"""Pre-fit presentation and confirmation callback.

Moves the config summary and interactive prompt out of BaseAgent
into a presentation-focused callback that runs at fit start.
"""

from __future__ import annotations

import pytorch_lightning as pl


class PrefitPresentationCallback(pl.Callback):
    """Display config summary and prompt to start training at fit start.

    If the user declines, sets trainer.should_stop and marks the module so that
    downstream teardown (e.g., videos/report) can choose to skip work.
    """

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        # Lazy imports to keep module load light
        from dataclasses import asdict
        from utils.logging import display_config_summary
        from utils.torch import _device_of
        from utils.user import prompt_confirm

        # Best-effort guards: require expected attributes on the module
        config = getattr(pl_module, "config", None)
        run = getattr(pl_module, "run", None)
        policy_model = getattr(pl_module, "policy_model", None)
        get_env = getattr(pl_module, "get_env", None)
        if not all([config, run, policy_model, callable(get_env)]):
            return

        # Compute model parameter counts
        try:
            num_params_total = int(sum(p.numel() for p in policy_model.parameters()))
            num_params_trainable = int(sum(p.numel() for p in policy_model.parameters() if p.requires_grad))
        except Exception:
            num_params_total = None
            num_params_trainable = None

        # Collect config and env details for presentation
        try:
            config_dict = asdict(config)
        except Exception:
            config_dict = {}

        try:
            train_env = get_env("train")
            env_block = {
                "env_id": getattr(train_env, "get_id", lambda: None)(),
                "obs_type": getattr(train_env, "get_obs_type", lambda: None)(),
                "obs_space": getattr(train_env, "observation_space", None),
                "action_space": getattr(train_env, "action_space", None),
                "reward_threshold": getattr(train_env, "get_reward_threshold", lambda: None)(),
                "time_limit": getattr(train_env, "get_time_limit", lambda: None)(),
                "env_wrappers": config_dict.get("env_wrappers", None),
                "env_kwargs": config_dict.get("env_kwargs", None),
            }
        except Exception:
            env_block = {}

        model_block = {
            "algo_id": getattr(config, "algo_id", None),
            "policy_class": type(policy_model).__name__ if policy_model is not None else None,
            "hidden_dims": getattr(config, "hidden_dims", None),
            "activation": getattr(config, "activation", None),
            "device": _device_of(policy_model) if policy_model is not None else None,
            "num_params_total": num_params_total,
            "num_params_trainable": num_params_trainable,
        }

        # Present summary
        display_config_summary({
            "Run": {
                "run_id": getattr(run, "get_run_id", lambda: None)(),
            },
            "Environment": env_block,
            "Model": model_block,
            "Config": config_dict,
        })

        # Optional guidance before prompting
        maybe_warn = getattr(pl_module, "_maybe_warn_observation_policy_mismatch", None)
        if callable(maybe_warn):
            try:
                maybe_warn()
            except Exception:
                pass

        # Confirm start
        quiet = bool(getattr(config, "quiet", False))
        start_training = prompt_confirm("Start training?", default=True, quiet=quiet)
        if not start_training:
            trainer.should_stop = True
            # Provide a clear stop reason and mark abort for downstream guards
            setattr(pl_module, "_early_stop_reason", "User aborted before training.")
            setattr(pl_module, "_aborted_before_training", True)
