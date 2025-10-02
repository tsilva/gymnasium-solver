"""Evaluate a trained Brax PPO policy from a saved checkpoint.

Usage:
  python scripts/brax_eval_policy.py --env inverted_pendulum --ckpt runs/brax-inverted_pendulum/<step> --episodes 5

If you just ran brax_train_policy.py, it prints the exact `--ckpt` path.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Tuple


def _soft_import_brax():
    try:
        import jax  # noqa: F401
        from brax import envs
        from brax.training.agents.ppo import checkpoint as ppo_ckpt
    except Exception as e:
        raise ImportError(
            "Missing Brax/JAX. Install GPU wheels: pip install -U \"jax[cuda12]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip install brax"
        ) from e
    return envs, ppo_ckpt


def _create_env(envs_mod, env_name: str, episode_length: Optional[int]):
    try:
        if episode_length is None:
            return envs_mod.create(env_name=env_name)
        return envs_mod.create(env_name=env_name, episode_length=episode_length)
    except TypeError:
        env = envs_mod.create(env_name=env_name)
        try:
            from brax.envs.wrappers import training as wrappers_training  # type: ignore

            return wrappers_training.wrap(env, episode_length=episode_length)
        except Exception:
            return env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Brax PPO policy checkpoint")
    p.add_argument("--env", default="inverted_pendulum")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="Brax checkpoint step directory (contains config.json and checkpoint data)")
    src.add_argument("--params", help="Flax params msgpack produced by brax_train_policy.py")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--episode_length", type=int, default=256)
    p.add_argument("--html", type=str, default=None, help="Optional path to save the first episode as an HTML viewer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    envs_mod, ppo_ckpt = _soft_import_brax()

    env = _create_env(envs_mod, args.env, args.episode_length)
    # Build policy either from a Brax checkpoint or from Flax params
    if args.ckpt:
        if not os.path.isdir(args.ckpt):
            print(f"Checkpoint dir not found: {args.ckpt}", file=sys.stderr)
            sys.exit(1)
        policy = ppo_ckpt.load_policy(args.ckpt, deterministic=True)
    else:
        # Load Flax params msgpack and construct networks/inference
        try:
            from flax import serialization as flax_serial
        except Exception:
            print("Flax not installed; cannot load params msgpack. Install flax.", file=sys.stderr)
            raise
        try:
            from brax.training.agents.ppo import networks as ppo_networks
        except Exception:
            print("Missing Brax PPO networks; upgrade brax.", file=sys.stderr)
            raise
        if not os.path.isfile(args.params):
            print(f"Params file not found: {args.params}", file=sys.stderr)
            sys.exit(1)
        raw = open(args.params, "rb").read()
        restored = flax_serial.msgpack_restore(raw)

        # Coerce to the tuple structure expected by make_inference_fn
        def _coerce_params_tuple(obj: Any) -> Tuple[Any, Any, Any]:
            # already a tuple/list
            if isinstance(obj, (list, tuple)):
                if len(obj) >= 3:
                    return (obj[0], obj[1], obj[2])
                if len(obj) == 2:
                    return (obj[0], obj[1], None)
            # dict with numeric keys
            if isinstance(obj, dict):
                if 0 in obj and 1 in obj:
                    return (obj[0], obj[1], obj.get(2))
                if "0" in obj and "1" in obj:
                    return (obj["0"], obj["1"], obj.get("2"))
                # common named keys
                n = obj.get("normalizer") or obj.get("normalizer_params") or {}
                p = obj.get("policy") or obj.get("policy_params")
                v = obj.get("value") or obj.get("value_params")
                if p is not None:
                    return (n, p, v)
                # last resort: order by sortable keys
                try:
                    keys = sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                    vals = [obj[k] for k in keys]
                    if len(vals) >= 2:
                        return (vals[0], vals[1], vals[2] if len(vals) > 2 else None)
                except Exception:
                    pass
            raise ValueError("Unrecognized params structure in msgpack; cannot make inference.")

        params_tuple = _coerce_params_tuple(restored)

        pnet = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
        )
        make_policy = ppo_networks.make_inference_fn(pnet)
        policy = make_policy(params_tuple, deterministic=True)

    # Roll out episodes and report returns
    import jax

    rng = jax.random.PRNGKey(0)
    total_rewards = []
    html_states = None
    for ep in range(args.episodes):
        rng, key_reset, key_action = jax.random.split(rng, 3)
        state = env.reset(key_reset)
        ep_reward = 0.0
        steps = 0
        ep_states = []
        for _ in range(args.episode_length):
            steps += 1
            key_action, subkey = jax.random.split(key_action)
            action, _ = policy(state.obs, subkey)
            state = env.step(state, action)
            # For HTML visualization, Brax expects a list of brax.base.State
            # with physics fields (q, qd, x, xd, contact). Convert from
            # the env state's pipeline_state.
            try:
                from brax import base as brax_base  # local import to avoid overhead
                ps = state.pipeline_state
                bs = brax_base.State(q=ps.q, qd=ps.qd, x=ps.x, xd=ps.xd, contact=getattr(ps, 'contact', None))
                ep_states.append(bs)
            except Exception:
                # Fallback: skip frame if conversion fails
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
    if args.html and html_states:
        try:
            from brax.io import html
            html.save(args.html, env.sys, html_states)
            print(f"Saved HTML rollout to: {args.html}")
        except Exception as e:
            print(f"Warning: failed to save HTML ({e!r})")


if __name__ == "__main__":
    main()
