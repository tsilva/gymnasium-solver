import torch
import numpy as np
from typing import Optional, Tuple, Sequence
from torch.utils.data import Dataset as TorchDataset
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from collections import deque

# TODO: is this needed?
def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

# TODO: add # env.normalize_obs() support
# TODO: batch value inference post rollout
# TODO: consider returning transition object
def collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    *,
    value_model: Optional[torch.nn.Module] = None,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    deterministic: bool = False,
    gamma: float = 0.99, # TODO: rename these?
    lam: float = 0.95, # TODO: rename these?
    normalize_advantage: bool = True, # TODO: should advantages be normalized during rollouts or during batch training?
    advantages_norm_eps: float = 1e-8,
    collect_frames: bool = False,
    last_obs: Optional[np.ndarray] = None # TODO: can I avoid using this by using a generator?
) -> Tuple[torch.Tensor, ...]:
    # Assert that a stop condition is provided (either n_steps or n_episodes)
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    # Assert that the models that are going to be used to perform 
    # the rollout are in eval mode (eg: no dropout, no batch norm updates)
    assert policy_model.training is False, "Policy model must be in eval mode for rollouts."
    if value_model: assert value_model.training is False, "Value model must be in eval mode for rollouts."

    # Assert that the policy model is on the same device as the value model
    policy_model_device = _device_of(policy_model)
    value_model_device = _device_of(value_model) if value_model else None
    if value_model_device: assert policy_model_device == value_model_device, "Policy and value models must be on the same device."

    device: torch.device = _device_of(policy_model)
    n_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # 1. Buffers (dynamic lists — we'll stack/concat later)
    # ------------------------------------------------------------------
    obs_buf: list[np.ndarray] = []
    actions_buf: list[np.ndarray] = []
    rewards_buf: list[np.ndarray] = []
    done_buf: list[np.ndarray] = []
    logprobs_buf: list[np.ndarray] = []
    values_buf: list[np.ndarray] = []
    frame_buf: list[Sequence[np.ndarray]] = []

    step_count = 0
    episode_count = 0 # why does this appear as a numpy value in the debugger?

    # ------------------------------------------------------------------
    # 2. Rollout
    # ------------------------------------------------------------------
    
    # Retrieve the first observation to start the rollout from, 
    # if last_obs is provided then it means we are resuming a previous rollout
    obs = env.reset() if last_obs is None else last_obs

    def _infer_value(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if value_model: return value_model(obs_t).squeeze(-1).cpu().numpy()
            else: return np.zeros((n_envs,), dtype=np.float32)

    def _infer_policy(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) # TODO: as_tensor vs tensor()
            logits = policy_model(obs_t)
            dist = Categorical(logits=logits)
            action_t = dist.sample() if not deterministic else logits.argmax(-1)
            lobprob_t = dist.log_prob(action_t)
            action_np = action_t.cpu().numpy()
            action_logp_np = lobprob_t.cpu().numpy()
            return action_np, action_logp_np

    # If last_obs is provided, we need to ensure it is on the correct device
    while True:
        # Use policy model to select action
        action_np, action_logp_np = _infer_policy(obs)

        # Use value model to infer value of being in current state
        value_np = _infer_value(obs)

        # Step the environment with the selected action
        next_obs, reward, done, _ = env.step(action_np)

        # Store collected data in buffers
        # NOTE: observation is copied to avoid issues with environments that mutate their observation arrays
        obs_buf.append(obs.copy())
        values_buf.append(value_np)
        actions_buf.append(action_np)
        logprobs_buf.append(action_logp_np)
        rewards_buf.append(reward)
        done_buf.append(done)
        if collect_frames: frame_buf.append(env.get_images())
        
        # Next state is now current state
        obs = next_obs

        # Increment the number of steps taken
        step_count += 1

        # Increment the number of episodes completed 
        # so far (add number of envs that are done)
        episode_count += done.sum()

        # If we reached a stop condition, pause the rollout
        if n_steps is not None and step_count >= n_steps: break # TODO: do we want the n_steps to be per env or total?
        if n_episodes is not None and episode_count >= n_episodes: break

    # ------------------------------------------------------------------
    # 3. Bootstrap value for the next state of each env
    # ------------------------------------------------------------------
    next_value = _infer_value(obs) # TODO: referencing next_obs could be less confusing

    # ------------------------------------------------------------------
    # 5. Stack buffers to (T, E) arrays for GAE
    # ------------------------------------------------------------------
    # TODO: not sure what this is
    obs_arr = np.stack(obs_buf)  # (T, E, obs)
    values_arr = np.stack(values_buf)
    actions_arr = np.stack(actions_buf)
    logprobs_arr = np.stack(logprobs_buf)
    rewards_arr = np.stack(rewards_buf)
    dones_arr = np.stack(done_buf)
    
    # ------------------------------------------------------------------
    # 5. GAE-λ advantage / return (with masking)
    # ------------------------------------------------------------------
    advantages_arr = np.zeros_like(rewards_arr, dtype=np.float32)

    gae = np.zeros(n_envs, dtype=np.float32)

    # Pre‑compute the mask that tells us whether each step is terminal (0) or not (1).
    non_terminal = 1.0 - dones_arr.astype(np.float32)

    # Start with the value *after* the last step (bootstrapped estimate) …
    next_non_terminal = non_terminal[-1]
    T = step_count # TODO: should step_count be the total and T be steps_count // n_envs?
    for t in reversed(range(T)):
        # Generalised Advantage Estimation update
        delta = rewards_arr[t] + gamma * next_value * next_non_terminal - values_arr[t]
        gae = delta + gamma * lam * gae * next_non_terminal 
        advantages_arr[t] = gae

        # Shift the window backwards so that in the next loop iteration
        # `next_value` / `next_non_terminal` correspond to step t‑1.
        next_value = values_arr[t]        # V(s_{t})
        next_non_terminal = non_terminal[t]

    # The actual returns, which is essentially the same as the
    # predicted returns plus the advantages (how much better 
    # or worse the returns were from the predicted)
    returns_arr = advantages_arr + values_arr # TODO: collect returns directly and then assert they match the computed ones

    if normalize_advantage:
        advantages_flat = advantages_arr.reshape(-1)
        advantages_arr = (advantages_arr - advantages_flat.mean()) / (advantages_flat.std() + advantages_norm_eps)

    # ------------------------------------------------------------------
    # 6. Env-major flattening: (T, E, …) -> (E, T, …) -> (E*T, …)
    # ------------------------------------------------------------------
    
    # TODO: need to understand what's going on here
    obs_env_major = obs_arr.transpose(1, 0, 2)  # (E, T, obs)
    states = torch.as_tensor(obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32)

    def _flat_env_major(arr: np.ndarray, dtype: torch.dtype):
        return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)

    actions = _flat_env_major(actions_arr, torch.int64)
    rewards = _flat_env_major(rewards_arr, torch.float32)
    dones = _flat_env_major(dones_arr, torch.bool)
    logps = _flat_env_major(logprobs_arr, torch.float32)
    values = _flat_env_major(values_arr, torch.float32)
    advantages = _flat_env_major(advantages_arr, torch.float32)
    returns = _flat_env_major(returns_arr, torch.float32)

    if collect_frames and frame_buf is not None:
        frames_env_major: list[np.ndarray] = []
        for e in range(n_envs):
            for t in range(T):
                frames_env_major.append(frame_buf[t][e])
    else:
        frames_env_major = [0] * (n_envs * T)

    return (
        states,
        actions,
        rewards,
        dones,
        logps,
        values,
        advantages,
        returns,
        frames_env_major,
    ), dict(last_obs=obs, n_steps=T, n_episodes=episode_count)


def group_trajectories_by_episode(trajectories, max_episodes=None):#
    """Group trajectory data by complete episodes.
    
    Args:
        trajectories: Tuple of trajectory tensors
        max_episodes: If specified, return only the first max_episodes complete episodes
        
    Returns:
        List of episodes, where each episode is a list of steps
    """
    episodes = []
    episode = []

    T = trajectories[0].shape[0]  # number of time steps

    for t in range(T):
        step = tuple(x[t] for x in trajectories)  # (state, action, reward, done, ...)
        episode.append(step)
        done = step[3]
        if done.item():  # convert tensor to bool
            episodes.append(episode)
            episode = []
            
            # Stop if we've collected enough complete episodes
            if max_episodes is not None and len(episodes) >= max_episodes:
                break

    return episodes

# TODO: add test script for collection using dataset/dataloader
# TODO: should dataloader move to gpu?
class RolloutDataset(TorchDataset):
    """Holds PPO roll-out tensors and lets them be swapped in-place."""
    def __init__(self):
        self.trajectories = None 

    def update(self, *trajectories):
        self.trajectories = trajectories

    def __len__(self):
        if self.trajectories is None: return 0
        return len(self.trajectories[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.trajectories)

# TODO: make rollout collector use its own buffer
class SyncRolloutCollector():
    def __init__(self, env, policy_model, value_model=None, n_steps=2048):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.n_steps = n_steps
        self.dataset = RolloutDataset()
        
        # TODO: move inside rollout collector?
        self.episode_reward_deque = deque(maxlen=100)#TODO: softcode config.mean_reward_window)

        self.last_obs = None

    def create_dataloader(self, batch_size: int = 64):
        return DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=True,
            # TODO: fix num_workers, training not converging when they are on
            # Pin memory is not supported on MPS
            #pin_memory=True if self.device.type != 'mps' else False,
            # TODO: Persistent workers + num_workers is fast but doesn't converge
            #persistent_workers=True if self.device.type != 'mps' else False,
            # Using multiple workers stalls the start of each epoch when persistent workers are disabled
            #num_workers=multiprocessing.cpu_count() // 2 if self.device.type != 'mps' else 0
        )
    
    # TODO: should this be called collect_rollout instead?
    # TODO: should this be a generator?
    # TODO: create decorator that ensures models are in eval mode until function end and then restores them to origial mode (eval or train)
    def collect_rollouts(self):
        self.policy_model.eval()
        if self.value_model: self.value_model.eval()
        try:
            trajectories, extras = collect_rollouts(
                self.env,
                self.policy_model,
                value_model=self.value_model,
                n_steps=self.n_steps,
                last_obs=self.last_obs
            )
            self.dataset.update(*trajectories)
            
            episodes = group_trajectories_by_episode(trajectories) # TODO: this is broken, as well as all other places that group per episode, because some episodes are lost in process
            episode_rewards = [sum(step[2] for step in episode) for episode in episodes]
            for r in episode_rewards: self.episode_reward_deque.append(float(r))

            self.last_obs = extras['last_obs'] # TODO: could I get rid of this if I used a generator?
            return trajectories
        finally:
            self.policy_model.train()
            if self.value_model: self.value_model.train()
