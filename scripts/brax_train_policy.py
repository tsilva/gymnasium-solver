"""Train and/or evaluate a simple policy on a Brax environment using PPO.

Minimal, self-contained example that depends on Brax/JAX. It does not
integrate with this repo's training loop; it's a standalone sample to
demonstrate Brax training end-to-end.

Usage examples:
- Train (defaults shown):
  python scripts/brax_train_policy.py --env inverted_pendulum --timesteps 200000
- Evaluate from params (HTML export):
  python scripts/brax_train_policy.py eval --env inverted_pendulum --params runs/brax-inverted_pendulum/<ts>/params.msgpack --episodes 1 --html runs/brax-inverted_pendulum/rollout.html
- Evaluate from Brax checkpoint (when available):
  python scripts/brax_train_policy.py eval --env inverted_pendulum --ckpt runs/brax-inverted_pendulum/<ts>/<step> --episodes 5

If Brax/JAX are missing, the script prints an install hint, e.g.:
  pip install -U "jax[cpu]" brax flax optax

Notes:
- The exact Brax API varies across versions. This script attempts to be
  robust by trying common import paths and parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


def _soft_import_brax():
    """Import brax/jax and resolve a PPO train function with clear errors.

    Returns: (jax, envs_module, ppo_train_callable, flax_serialization_or_None)
    """
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # noqa: F401  # type: ignore
    except Exception as e:  # pragma: no cover - soft dep
        raise ImportError(
            "JAX is required for Brax. Install with: pip install -U \"jax[cpu]\""
        ) from e

    try:
        import brax  # noqa: F401  # type: ignore
        from brax import envs  # type: ignore
    except Exception as e:  # pragma: no cover - soft dep
        raise ImportError(
            "Brax is not installed. Install with: pip install -U brax"
        ) from e

    # Resolve a PPO train function across Brax versions
    ppo_train = None
    # Newer flat path sometimes exposes brax.training.ppo.train
    try:
        from brax.training import ppo as _ppo_pkg  # type: ignore
        ppo_train = getattr(_ppo_pkg, "train", None)
    except Exception:
        ppo_train = None
    if ppo_train is None:
        # v0.13 structure: brax.training.agents.ppo.train.train
        try:
            from brax.training.agents.ppo.train import train as ppo_train  # type: ignore
        except Exception:
            try:
                # Some builds allow package import then attribute
                from brax.training.agents import ppo as _ppo_pkg  # type: ignore
                ppo_train = getattr(_ppo_pkg, "train", None)
            except Exception:
                ppo_train = None
    if ppo_train is None:
        raise ImportError(
            "Brax PPO training API not found. Try upgrading brax: pip install -U brax"
        )

    # Flax for saving params is optional
    try:
        from flax import serialization as flax_serial  # type: ignore
    except Exception:  # pragma: no cover - soft dep
        flax_serial = None  # type: ignore

    return jax, envs, ppo_train, flax_serial


def _available_env_names(envs_mod) -> Tuple[str, ...]:
    names = ()
    try:
        registry = getattr(envs_mod, "_envs", None)
        if registry:
            names = tuple(sorted(registry.keys()))
    except Exception:
        pass
    if not names:
        try:
            registry = getattr(envs_mod, "ENVS", None)
            if registry:
                names = tuple(sorted(registry.keys()))
        except Exception:
            pass
    return names


def _create_env(envs_mod, env_name: str, episode_length: Optional[int]):
    """Create a Brax environment with optional episode length.

    Supports common env creation paths across Brax versions.
    """
    # Newer API often uses envs.create with episode_length; fall back as needed.
    try:
        if episode_length is None:
            return envs_mod.create(env_name=env_name)
        return envs_mod.create(env_name=env_name, episode_length=episode_length)
    except KeyError as e:
        # Helpful message with available names/suggestions
        names = _available_env_names(envs_mod)
        hint = " | ".join([n for n in names if any(k in n for k in ("pendulum","ant","cheetah","reacher","hopper","walker"))])
        msg = [f"Unknown Brax env: {env_name!r}."]
        if names:
            msg.append(f"Available: {', '.join(names)}")
        if hint:
            msg.append(f"Try something like: {hint}")
        raise KeyError(" \n".join(msg)) from e
    except TypeError:
        # Some versions don't accept episode_length here; create then wrap if available.
        env = envs_mod.create(env_name=env_name)
        try:
            # In some versions, training.wrap is under envs.wrappers.training
            from brax.envs.wrappers import training as wrappers_training  # type: ignore

            return wrappers_training.wrap(env, episode_length=episode_length)
        except Exception:
            return env
    except AttributeError:
        # Older variants may expose get_environment
        env = envs_mod.get_environment(env_name=env_name)
        return env


@dataclass
class BraxTrainConfig:
    env: str = "inverted_pendulum"
    seed: int = 0
    timesteps: int = 200_000
    episode_length: int = 256
    num_envs: int = 1024
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-3
    discounting: float = 0.97
    unroll_length: int = 10
    batch_size: int = 1024
    outdir: Optional[str] = None


def parse_args() -> BraxTrainConfig:
    p = argparse.ArgumentParser(
        description="Train a PPO policy on a Brax environment",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train a simple policy (default mode)\n"
            "  python scripts/brax_train_policy.py --env inverted_pendulum --timesteps 200000\n\n"
            "  # Evaluate a saved policy with HTML export\n"
            "  python scripts/brax_train_policy.py eval \\\n+            \n    --env inverted_pendulum \\\n+            \n    --params runs/brax-inverted_pendulum/<ts>/params.msgpack \\\n+            \n    --episodes 1 --html runs/brax-inverted_pendulum/rollout.html\n"
        ),
    )
    p.add_argument("--env", default="inverted_pendulum", help="Brax env name, e.g. inverted_pendulum, reacher, ant, halfcheetah")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--episode_length", type=int, default=256)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--entropy_cost", type=float, default=1e-3)
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--unroll_length", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--outdir", type=str, default=None, help="Optional dir to save params/metrics (defaults under runs/)")
    args = p.parse_args()
    return BraxTrainConfig(
        env=args.env,
        seed=args.seed,
        timesteps=args.timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        outdir=args.outdir,
    )


def _default_outdir(cfg: BraxTrainConfig) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join("runs", f"brax-{cfg.env}")
    return os.path.join(base, ts)


def _train(cfg: BraxTrainConfig, *, save_ckpt_dir: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    jax, envs_mod, ppo_train, _ = _soft_import_brax()
    env = _create_env(envs_mod, cfg.env, cfg.episode_length)

    # ppo.train signature is mostly stable; pass common args only.
    train_kwargs: Dict[str, Any] = dict(
        environment=env,
        num_timesteps=int(cfg.timesteps),
        episode_length=int(cfg.episode_length),
        num_envs=int(cfg.num_envs),
        learning_rate=float(cfg.learning_rate),
        entropy_cost=float(cfg.entropy_cost),
        discounting=float(cfg.discounting),
        seed=int(cfg.seed),
    )
    # Optional args vary by version; include when available
    for k in ("unroll_length", "batch_size"):
        train_kwargs[k] = int(getattr(cfg, k))

    # Save Brax-native checkpoints if a path is provided
    if save_ckpt_dir:
        train_kwargs["save_checkpoint_path"] = save_ckpt_dir

    # Run training (JIT-compiles on first call; this can take time)
    inference_fn, params, metrics = ppo_train(**train_kwargs)

    # Gather a compact summary of the last metrics for visibility
    # Common keys: 'eval/episode_reward', 'train/episode_reward', etc.
    last_metrics = {k: (v[-1] if isinstance(v, (list, tuple)) else v) for k, v in metrics.items()}
    print("\n=== Training complete ===")
    for k in sorted(last_metrics.keys()):
        try:
            print(f"{k}: {float(last_metrics[k]):.4f}")
        except Exception:
            print(f"{k}: {last_metrics[k]}")

    return inference_fn, params, metrics


def _save(outdir: str, cfg: BraxTrainConfig, params: Any, metrics: Dict[str, Any]) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Save config and metrics as JSON
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    # Convert non-serializable metrics to floats when possible
    m_serializable: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, (list, tuple)):
                m_serializable[k] = [float(x) for x in v]
            else:
                m_serializable[k] = float(v)
        except Exception:
            m_serializable[k] = str(v)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(m_serializable, f, indent=2)

    # Attempt to save parameters using Flax serialization when available
    try:
        _, _, _, flax_serial = _soft_import_brax()
    except Exception:
        flax_serial = None
    if flax_serial is not None:
        try:
            bytestr = flax_serial.to_bytes(params)
            with open(os.path.join(outdir, "params.msgpack"), "wb") as f:
                f.write(bytestr)
            print(f"Saved params to {os.path.join(outdir, 'params.msgpack')}")
        except Exception as e:
            print(f"Warning: failed to serialize params with Flax ({e!r}). Skipping.")
    else:
        print("Flax not available; skipping params serialization.")


# =====================
# Evaluation (subcommand)
# =====================

def _load_policy_from_ckpt(env, ckpt_dir: str):
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    return ppo_ckpt.load_policy(ckpt_dir, deterministic=True)


def _coerce_params_tuple(obj: Any) -> Tuple[Any, Any, Any]:
    # already a tuple/list
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 3:
            return (obj[0], obj[1], obj[2])
        if len(obj) == 2:
            return (obj[0], obj[1], None)
    # dict with numeric or named keys
    if isinstance(obj, dict):
        if 0 in obj and 1 in obj:
            return (obj[0], obj[1], obj.get(2))
        if "0" in obj and "1" in obj:
            return (obj["0"], obj["1"], obj.get("2"))
        n = obj.get("normalizer") or obj.get("normalizer_params") or {}
        p = obj.get("policy") or obj.get("policy_params")
        v = obj.get("value") or obj.get("value_params")
        if p is not None:
            return (n, p, v)
        try:
            keys = sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
            vals = [obj[k] for k in keys]
            if len(vals) >= 2:
                return (vals[0], vals[1], vals[2] if len(vals) > 2 else None)
        except Exception:
            pass
    raise ValueError("Unrecognized params structure in msgpack; cannot make inference.")


def _load_policy_from_params(env, params_path: str):
    from flax import serialization as flax_serial
    from brax.training.agents.ppo import networks as ppo_networks
    raw = open(params_path, "rb").read()
    restored = flax_serial.msgpack_restore(raw)
    params_tuple = _coerce_params_tuple(restored)
    pnet = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(pnet)
    return make_policy(params_tuple, deterministic=True)


def _eval_rollout(
    env_name: str,
    *,
    ckpt_dir: Optional[str] = None,
    params_path: Optional[str] = None,
    episodes: int = 5,
    episode_length: int = 256,
    html_path: Optional[str] = None,
) -> None:
    jax, envs_mod, _, _ = _soft_import_brax()
    env = _create_env(envs_mod, env_name, episode_length)

    if ckpt_dir:
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
        policy = _load_policy_from_ckpt(env, ckpt_dir)
    else:
        if not params_path or not os.path.isfile(params_path):
            raise FileNotFoundError(f"Params file not found: {params_path}")
        policy = _load_policy_from_params(env, params_path)

    rng = jax.random.PRNGKey(0)
    total_rewards = []
    html_states = None
    for ep in range(episodes):
        rng, key_reset, key_action = jax.random.split(rng, 3)
        state = env.reset(key_reset)
        ep_reward = 0.0
        steps = 0
        ep_states = []
        for _ in range(episode_length):
            steps += 1
            key_action, subkey = jax.random.split(key_action)
            action, _ = policy(state.obs, subkey)
            state = env.step(state, action)
            # Convert to brax.base.State for HTML saving
            try:
                from brax import base as brax_base
                ps = state.pipeline_state
                bs = brax_base.State(q=ps.q, qd=ps.qd, x=ps.x, xd=ps.xd, contact=getattr(ps, 'contact', None))
                ep_states.append(bs)
            except Exception:
                pass
            ep_reward += float(state.reward)
            if bool(state.done):
                break
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, steps={steps}")
        if html_states is None:
            html_states = ep_states

    avg = sum(total_rewards) / max(1, len(total_rewards))
    print(f"Average reward over {len(total_rewards)} episodes: {avg:.2f}")

    if html_path and html_states:
        try:
            from brax.io import html
            html.save(html_path, env.sys, html_states)
            print(f"Saved HTML rollout to: {html_path}")
        except Exception as e:
            print(f"Warning: failed to save HTML ({e!r})")


def _find_latest_ckpt_dir(base_dir: str) -> Optional[str]:
    try:
        entries = [d for d in os.listdir(base_dir) if d.isdigit()]
        if not entries:
            return None
        latest = sorted(entries)[-1]
        return os.path.join(base_dir, latest)
    except Exception:
        return None


def _supports_brax_checkpoint() -> bool:
    """Return True if current JAX/Brax/Orbax stack can save checkpoints.

    Brax's checkpoint path relies on orbax + jax.monitoring.record_scalar in
    recent JAX. If the attribute is missing, saving will raise.
    """
    try:
        import jax  # type: ignore
        from orbax import checkpoint as ocp  # noqa: F401
        return hasattr(jax, "monitoring") and hasattr(jax.monitoring, "record_scalar")
    except Exception:
        return False


def _parse_eval_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(
        prog="brax_train_policy.py eval",
        description="Evaluate a Brax PPO policy",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Evaluate from Flax params\n"
            "  python scripts/brax_train_policy.py eval \\\n+            \n    --env inverted_pendulum \\\n+            \n    --params runs/brax-inverted_pendulum/<ts>/params.msgpack --episodes 5\n\n"
            "  # Evaluate from a Brax checkpoint (when available)\n"
            "  python scripts/brax_train_policy.py eval \\\n+            \n    --env inverted_pendulum \\\n+            \n    --ckpt runs/brax-inverted_pendulum/<ts>/<step> --episodes 5\n"
        ),
    )
    p.add_argument("--env", default="inverted_pendulum")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="Brax checkpoint step directory (contains config.json and checkpoint data)")
    src.add_argument("--params", help="Flax params msgpack produced by training")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--episode_length", type=int, default=256)
    p.add_argument("--html", type=str, default=None, help="Optional path to save the first episode as an HTML viewer")
    return p.parse_args(argv)


def main() -> None:
    # Subcommand support: `eval` and `train` (default)
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        args = _parse_eval_args(sys.argv[2:])
        _eval_rollout(
            args.env,
            ckpt_dir=args.ckpt,
            params_path=args.params,
            episodes=args.episodes,
            episode_length=args.episode_length,
            html_path=args.html,
        )
        return

    # Default: train mode (backwards compatible flags)
    cfg = parse_args()
    outdir = cfg.outdir or _default_outdir(cfg)
    os.makedirs(outdir, exist_ok=True)
    try:
        save_ckpt = outdir if _supports_brax_checkpoint() else None
        if save_ckpt is None:
            print("Note: Brax checkpoint disabled (incompatible JAX/orbax). Using Flax params only.")
        _, params, metrics = _train(cfg, save_ckpt_dir=save_ckpt)
    except ImportError as e:
        # Provide actionable guidance for missing soft deps
        print(str(e), file=sys.stderr)
        print(
            "\nHint: install CPU-only deps with:\n  pip install -U \"jax[cpu]\" brax flax optax\n",
            file=sys.stderr,
        )
        sys.exit(1)

    _save(outdir, cfg, params, metrics)
    ckpt_dir = _find_latest_ckpt_dir(outdir)
    if ckpt_dir:
        print(f"Saved Brax checkpoint to: {ckpt_dir}")
        print("Evaluate it with:")
        print(f"  python scripts/brax_train_policy.py eval --env {cfg.env} --ckpt {ckpt_dir} --episodes 5")
    else:
        print("Brax checkpoint not available; evaluate with Flax params:")
        params_path = os.path.join(outdir, 'params.msgpack')
        html_path = os.path.join(outdir, 'rollout.html')
        print(f"  python scripts/brax_train_policy.py eval --env {cfg.env} --params {params_path} --episodes 5 --html {html_path}")
    print(f"Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
