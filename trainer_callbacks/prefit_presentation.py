"""Callback to present a config summary and confirm training start."""

from __future__ import annotations

import pytorch_lightning as pl

# TODO: REFACTOR this file

# TODO: call this callback something else
class PrefitPresentationCallback(pl.Callback):
    """Show a config summary at fit start and prompt to continue."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        # Lazy imports to keep module load light
        from dataclasses import asdict

        from utils.logging import display_config_summary
        from utils.torch import _device_of
        from utils.user import prompt_confirm

        # Best-effort guards: require expected attributes on the module
        config = pl_module.config
        run = pl_module.run
        policy_model = pl_module.policy_model
        get_env = pl_module.get_env
        assert all([config, run, policy_model, callable(get_env)])

        # Compute model parameter counts
        num_params_total = int(sum(p.numel() for p in policy_model.parameters()))
        num_params_trainable = int(sum(p.numel() for p in policy_model.parameters() if p.requires_grad))

        # Collect config and env details for presentation
        config_dict = asdict(config)
       
        train_env = get_env("train")
        env_block = {
            "env_id": train_env.get_id(),
            "obs_type": train_env.get_obs_type(),
            "obs_space": train_env.observation_space,
            "action_space": train_env.action_space,
            "reward_threshold": train_env.get_reward_threshold(),
            "time_limit": train_env.get_time_limit(),
            "env_wrappers": config_dict.get("env_wrappers", None),
            "env_kwargs": config_dict.get("env_kwargs", None),
        }

        model_block = {
            "algo_id": config.algo_id,
            "policy_class": type(policy_model).__name__ if policy_model is not None else None,
            "hidden_dims": config.hidden_dims,
            "activation": config.activation,
            "device": _device_of(policy_model) if policy_model is not None else None,
            "num_params_total": num_params_total,
            "num_params_trainable": num_params_trainable,
        }

        # Present summary
        display_config_summary({
            "Run": {
                "run_id": run.id,
            },
            "Environment": env_block,
            "Model": model_block,
            "Config": config_dict,
        })

        # Optional guidance before prompting
        maybe_warn = pl_module._maybe_warn_observation_policy_mismatch
        if callable(maybe_warn):
            try:
                maybe_warn()
            except Exception:
                pass

        # Confirm start
        quiet = bool(config.quiet)
        start_training = prompt_confirm("Start training?", default=True, quiet=quiet)
        if not start_training:
            trainer.should_stop = True
            # Provide a clear stop reason and mark abort for downstream guards
            pl_module._early_stop_reason = "User aborted before training."
            pl_module._aborted_before_training = True
