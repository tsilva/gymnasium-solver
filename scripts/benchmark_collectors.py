"""Benchmark RolloutCollector throughput."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.environment import build_env
from utils.models import MLPActorCritic
from utils.rollouts import RolloutCollector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RolloutCollector")
    parser.add_argument("--config", default="CartPole-v1:ppo", help="Config ID (env:variant)")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(*args.config.split(":"))
    vec_env = build_env(cfg.env_id, seed=cfg.seed, n_envs=cfg.n_envs, subproc=getattr(cfg, "subproc", None),
                        project_id=cfg.project_id, spec=cfg.spec, obs_type=cfg.obs_type,
                        env_wrappers=cfg.env_wrappers, normalize_obs=cfg.normalize_obs,
                        frame_stack=cfg.frame_stack, env_kwargs=cfg.env_kwargs)

    hidden = cfg.hidden_dims if isinstance(cfg.hidden_dims, (list, tuple)) else (cfg.hidden_dims,)
    policy = MLPActorCritic(vec_env.observation_space.shape[0], vec_env.action_space.n, hidden_dims=hidden)
    collector = RolloutCollector(vec_env, policy, n_steps=cfg.n_steps, **cfg.rollout_collector_hyperparams())

    for _ in range(args.warmup):
        collector.collect()

    start = time.perf_counter()
    for _ in range(args.rollouts):
        collector.collect()
    elapsed = time.perf_counter() - start

    steps = args.rollouts * cfg.n_envs * cfg.n_steps
    print(f"{steps / elapsed:,.0f} FPS | {steps:,} steps | {elapsed:.2f}s")
