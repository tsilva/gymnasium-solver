#!/usr/bin/env python3

import argparse
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Minari behavioral cloning example: load a dataset, train a simple policy, and render a demo run."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/door/human-v2",
        help=(
            "Minari dataset id to load. Defaults to a small D4RL dataset. "
            "Use `minari list remote` to view options."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--render", action="store_true", help="Render a demo rollout after training."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of demo episodes to run after training.",
    )
    return parser.parse_args()


def concatenate_episodes_to_arrays(dataset) -> Tuple[np.ndarray, np.ndarray]:
    observations_list = []
    actions_list = []
    for episode in dataset.iterate_episodes():
        # Episode arrays typically have shapes [T, obs_dim] and [T, action_dim or scalar].
        # Some datasets include an extra terminal observation (length T+1). Align by trimming to min length.
        obs = np.asarray(episode.observations)
        act = np.asarray(episode.actions)
        if obs.shape[0] != act.shape[0]:
            min_len = min(obs.shape[0], act.shape[0])
            obs = obs[:min_len]
            act = act[:min_len]
        observations_list.append(obs)
        actions_list.append(act)
    observations_array = np.concatenate(observations_list, axis=0)
    actions_array = np.concatenate(actions_list, axis=0)
    return observations_array, actions_array


class MLPPolicyContinuous(nn.Module):
    def __init__(self, input_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, int(action_low.shape[0])),
        )
        # Precompute scaling to map tanh output to action bounds
        self.register_buffer("action_scale", torch.from_numpy((action_high - action_low) / 2.0).float())
        self.register_buffer("action_bias", torch.from_numpy((action_high + action_low) / 2.0).float())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self.network(observations)
        bounded = torch.tanh(raw) * self.action_scale + self.action_bias
        return bounded


class MLPPolicyDiscrete(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def main() -> None:
    args = parse_args()

    try:
        import minari  # noqa: F401
    except ImportError:
        print(
            "Minari is required. Install with: pip install \"minari[all]\"",
            file=sys.stderr,
        )
        raise

    import gymnasium as gym
    import minari

    print(f"Loading Minari dataset: {args.dataset_id}")
    dataset = minari.load_dataset(args.dataset_id, download=True)

    # Determine observation/action spaces from dataset if available; otherwise from recovered env
    obs_space = getattr(dataset, "observation_space", None)
    act_space = getattr(dataset, "action_space", None)

    # Load arrays for BC
    observations_np, actions_np = concatenate_episodes_to_arrays(dataset)

    # Convert observations to 2D [N, obs_dim]
    if observations_np.ndim > 2:
        # Flatten everything except batch dimension (works for dict-free, array obs)
        observations_np = observations_np.reshape(observations_np.shape[0], -1)

    # Build tensors
    observations = torch.from_numpy(observations_np).float()

    # Build a policy and loss depending on action space type
    is_discrete = False
    if act_space is None:
        # Fallback: recover env just for spaces
        temp_env = dataset.recover_environment()
        act_space = temp_env.action_space
        obs_space = temp_env.observation_space if obs_space is None else obs_space
        temp_env.close()

    if hasattr(act_space, "n"):
        is_discrete = True
        num_actions = int(act_space.n)
        # Actions expected as class indices [N]
        if actions_np.ndim > 1:
            actions_np = actions_np.reshape(actions_np.shape[0], -1)
            if actions_np.shape[1] == 1:
                actions_np = actions_np[:, 0]
        actions = torch.from_numpy(actions_np).long()
        policy = MLPPolicyDiscrete(input_dim=observations.shape[1], num_actions=num_actions)
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        # Continuous Box
        action_low = np.array(act_space.low, dtype=np.float32).reshape(-1)
        action_high = np.array(act_space.high, dtype=np.float32).reshape(-1)
        # Ensure actions are 2D [N, action_dim]
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(-1, 1)
        actions = torch.from_numpy(actions_np).float()
        policy = MLPPolicyContinuous(
            input_dim=observations.shape[1], action_low=action_low, action_high=action_high
        )
        criterion = nn.MSELoss()

    # Final safety alignment: ensure observations and actions have the same number of samples
    final_len = min(observations.shape[0], actions.shape[0])
    if observations.shape[0] != final_len:
        observations = observations[:final_len]
    if actions.shape[0] != final_len:
        actions = actions[:final_len]

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Simple training loop (behavioral cloning)
    num_samples = observations.shape[0]
    indices = torch.arange(num_samples)
    batch_size = max(1, min(args.batch_size, num_samples))

    print(
        f"Training BC policy on {num_samples} samples for {args.epochs} epochs, batch_size={batch_size}"
    )
    policy.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, num_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_obs = observations[batch_idx]
            batch_act = actions[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            logits_or_actions = policy(batch_obs)
            if is_discrete:
                loss = criterion(logits_or_actions, batch_act)
            else:
                loss = criterion(logits_or_actions, batch_act)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            num_batches += 1
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss={avg_loss:.4f}")

    # Save model next to script for quick reuse
    save_path = "scripts/minari_bc_example.pt"
    torch.save({"state_dict": policy.state_dict(), "is_discrete": is_discrete}, save_path)
    print(f"Saved trained policy to: {save_path}")

    if not args.render:
        print("Skipping demo rollout (pass --render to show a window, if supported).")
        return

    # Recover the environment; try to enable human rendering.
    env = None
    try:
        env = dataset.recover_environment(render_mode="human")
    except TypeError:
        # Some Minari versions may not accept kwargs here.
        env = dataset.recover_environment()

    # Fallback: try gym.make using dataset metadata if recover_environment doesn't support human rendering
    if env is None or getattr(env, "render_mode", None) != "human":
        env_name = getattr(dataset, "environment_name", None)
        if env_name is not None:
            env = gym.make(env_name, render_mode="human")
    if env is None:
        print(
            "Could not create a human-rendering environment. Run without --render or try a different dataset.",
            file=sys.stderr,
        )
        return

    policy.eval()
    demo_episodes = max(1, args.episodes)
    for ep in range(demo_episodes):
        observation, _ = env.reset()
        # Flatten if needed for the policy
        while True:
            obs_array = np.asarray(observation)
            if obs_array.ndim > 1:
                obs_array = obs_array.reshape(-1)
            obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
            with torch.no_grad():
                logits_or_actions = policy(obs_tensor)
                if is_discrete:
                    action = int(torch.argmax(logits_or_actions, dim=-1).item())
                else:
                    action = logits_or_actions.squeeze(0).cpu().numpy()
            step_out = env.step(action)
            if len(step_out) == 5:
                observation, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                observation, reward, done, _ = step_out  # older API fallback
            # For human mode, some envs render automatically; keep explicit call for safety
            env.render()
            if done:
                break
    env.close()


if __name__ == "__main__":
    main()

