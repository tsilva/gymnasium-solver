"""Benchmark RolloutCollector throughput."""
import argparse
import sys
import time
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.environment import build_env
from utils.policy_factory import build_policy_from_env_and_config
from utils.rollouts import RolloutCollector


if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser(description="Benchmark RolloutCollector")
    parser.add_argument("--config", default="ALE/Pong-v5:rgb_ppo", help="Config ID (env:variant)")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    # Load config
    cfg = load_config(*args.config.split(":"))
    vec_env = build_env(
        cfg.env_id,
        seed=cfg.seed, 
        n_envs=cfg.n_envs, 
        subproc=getattr(cfg, "subproc", None),
        project_id=cfg.project_id, 
        spec=cfg.spec, 
        obs_type=cfg.obs_type,
        env_wrappers=cfg.env_wrappers, 
        normalize_obs=cfg.normalize_obs,
        frame_stack=cfg.frame_stack, 
        env_kwargs=cfg.env_kwargs
    )
   
    # Build policy
    policy = build_policy_from_env_and_config(vec_env, cfg)

    # Build collector
    collector = RolloutCollector(
        vec_env, 
        policy, 
        n_steps=cfg.n_steps, 
        **cfg.rollout_collector_hyperparams()
    )

    # Warmup
    for _ in range(args.warmup):
        collector.collect()

    # Benchmark
    start = time.perf_counter()
    for _ in tqdm(range(args.rollouts)):
        collector.collect()
    elapsed = time.perf_counter() - start

    # Print results
    steps = args.rollouts * cfg.n_envs * cfg.n_steps
    print(f"{steps / elapsed:,.0f} FPS | {steps:,} steps | {elapsed:.2f}s")
