import torch
import numpy as np
from typing import Optional, Tuple, Sequence
from torch.utils.data import Dataset as TorchDataset
from torch.distributions import Categorical
import copy
import time
import threading
import queue


def collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    value_model: Optional[torch.nn.Module] = None,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    *,
    deterministic: bool = False,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize_advantage: bool = True,
    adv_norm_eps: float = 1e-8,
    collect_frames: bool = False,
    last_obs: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, ...]:
    """Collect transitions from *env* until *n_steps* or *n_episodes* (whichever
    comes first) are reached.

    IMPORTANT: When n_episodes is specified, this function may collect slightly more
    episodes than requested due to vectorized environments. Use group_trajectories_by_episode()
    with max_episodes parameter to get exactly the desired number of complete episodes.

    Parameters
    ----------
    env : VecEnv-like
        Vectorised environment exposing ``reset()``, ``step(actions)`` and
        ``get_images()`` (if *collect_frames* is ``True``).
    policy_model : nn.Module
        Model that maps observations to *logits* (action probabilities).
    value_model : nn.Module | None
        Optional value network; if ``None``, value estimates are *zero*.
    n_steps : int | None, default 1024
        Maximum number of *timesteps* (across all environments) to collect.
    n_episodes : int | None, default None
        Maximum number of *episodes* (across all environments) to collect.
        May collect slightly more due to vectorized environments.
        Either *n_steps*, *n_episodes* or both **must** be provided.
    deterministic : bool, default False
        Whether to act greedily (``argmax``) instead of sampling.
    gamma, lam : float
        Discount and GAE-λ parameters.
    normalize_advantage : bool, default True
        Whether to standardise advantages.
    adv_norm_eps : float, default 1e-8
        Numerical stability epsilon for advantage normalisation.
    collect_frames : bool, default False
        If ``True``, return RGB frames alongside transition tensors.
    last_obs : np.ndarray | None, default None
        If provided, use these observations to continue collection without
        resetting the environment. If ``None``, reset the environment first.

    Returns
    -------
    Tuple[Tensor, ...]
        ``(states, actions, rewards, dones, logps, values, advs, returns, frames)``
        in *Stable-Baselines3*-compatible, env-major flattened order.
    """

    # ------------------------------------------------------------------
    # 0. Sanity checks & helpers
    # ------------------------------------------------------------------
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    def _device_of(module: torch.nn.Module) -> torch.device:
        """Infer the device of *module*'s first parameter."""
        # Handle shared backbone wrapper models
        if hasattr(module, 'shared_net'):
            return next(module.shared_net.parameters()).device
        return next(module.parameters()).device

    device: torch.device = _device_of(policy_model)
    n_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # 1. Buffers (dynamic lists — we'll stack/concat later)
    # ------------------------------------------------------------------
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    rew_buf: list[np.ndarray] = []
    done_buf: list[np.ndarray] = []
    logp_buf: list[np.ndarray] = []
    val_buf: list[np.ndarray] = []
    frame_buf: list[Sequence[np.ndarray]] | None = [] if collect_frames else None

    step_count = 0
    episode_count = 0

    # ------------------------------------------------------------------
    # 2. Rollout
    # ------------------------------------------------------------------
    obs = env.reset() if last_obs is None else last_obs
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = policy_model(obs_t)
        dist = Categorical(logits=logits)

        act_t = logits.argmax(-1) if deterministic else dist.sample()
        logp_t = dist.log_prob(act_t)
        val_t = (
            value_model(obs_t).squeeze(-1)
            if value_model is not None
            else torch.zeros(n_envs, device=device)
        )

        next_obs, reward, done, infos = env.step(act_t.cpu().numpy())

        # store step
        obs_buf.append(obs.copy())
        act_buf.append(act_t.cpu().numpy())
        rew_buf.append(reward)
        done_buf.append(done)
        logp_buf.append(logp_t.detach().cpu().numpy())
        val_buf.append(val_t.detach().cpu().numpy())

        if collect_frames and frame_buf is not None:
            frame_buf.append(env.get_images())

        step_count += 1
        episode_count += done.sum()

        # IMPROVED TERMINATION LOGIC: When using n_episodes, collect a bit more to ensure
        # we get at least n_episodes complete episodes. The exact truncation will be
        # handled by the caller if needed.
        should_stop = False
        if n_steps is not None and step_count >= n_steps:
            should_stop = True
        elif n_episodes is not None and episode_count >= n_episodes:
            should_stop = True
        
        if should_stop:
            obs = next_obs  # needed for bootstrap
            break

        obs = next_obs

    T = step_count  # actual collected timesteps

    # ------------------------------------------------------------------
    # 3. Bootstrap value for the next state of each env
    # ------------------------------------------------------------------
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        next_values = (
            value_model(obs_t).squeeze(-1).cpu().numpy()
            if value_model is not None
            else np.zeros(n_envs, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # 5. Stack buffers to (T, E) arrays for GAE
    # ------------------------------------------------------------------
    act_arr = np.stack(act_buf)  # (T, E)
    rew_arr = np.stack(rew_buf)
    done_arr = np.stack(done_buf)
    logp_arr = np.stack(logp_buf)
    val_arr = np.stack(val_buf)

    # ------------------------------------------------------------------
    # 5. GAE-λ advantage / return (with masking)
    # ------------------------------------------------------------------
    adv_arr = np.zeros_like(rew_arr, dtype=np.float32)

    gae = np.zeros(n_envs, dtype=np.float32)
    next_non_terminal = 1.0 - done_arr[-1].astype(np.float32)
    next_value = next_values

    for t in reversed(range(T)):
        delta = rew_arr[t] + gamma * next_value * next_non_terminal - val_arr[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        adv_arr[t] = gae

        next_non_terminal = 1.0 - done_arr[t].astype(np.float32)
        next_value = val_arr[t]

    ret_arr = adv_arr + val_arr

    if normalize_advantage:
        adv_flat = adv_arr.reshape(-1)
        adv_arr = (adv_arr - adv_flat.mean()) / (adv_flat.std() + adv_norm_eps)

    # ------------------------------------------------------------------
    # 6. Env-major flattening: (T, E, …) -> (E, T, …) -> (E*T, …)
    # ------------------------------------------------------------------
    obs_arr = np.stack(obs_buf)  # (T, E, obs)
    obs_env_major = obs_arr.transpose(1, 0, 2)  # (E, T, obs)
    states = torch.as_tensor(obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32)

    def _flat_env_major(arr: np.ndarray, dtype: torch.dtype):
        return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)

    actions = _flat_env_major(act_arr, torch.int64)
    rewards = _flat_env_major(rew_arr, torch.float32)
    dones = _flat_env_major(done_arr, torch.bool)
    logps = _flat_env_major(logp_arr, torch.float32)
    values = _flat_env_major(val_arr, torch.float32)
    advs = _flat_env_major(adv_arr, torch.float32)
    returns = _flat_env_major(ret_arr, torch.float32)

    if collect_frames and frame_buf is not None:
        frames_env_major: list[np.ndarray] = []
        for e in range(n_envs):
            for t in range(T):
                frames_env_major.append(frame_buf[t][e])
    else:
        frames_env_major = [0] * (n_envs * T)

    # ------------------------------------------------------------------
    # 7. Return (SB3 order) - Use group_trajectories_by_episode(max_episodes=n) for exact counts
    # ------------------------------------------------------------------
    return (
        states,
        actions,
        rewards,
        dones,
        logps,
        values,
        advs,
        returns,
        frames_env_major,
    ), dict(last_obs=obs, n_steps=T, n_episodes=episode_count)


def group_trajectories_by_episode(trajectories, max_episodes=None):
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



# TODO: should dataloader move to gpu?
class RolloutDataset(TorchDataset):
    """Holds PPO roll-out tensors and lets them be swapped in-place."""
    def __init__(self):
        self.trajectories = None 

    def update(self, *trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.trajectories)



class BaseRolloutCollector:
    """Base class for rollout collectors"""
    def __init__(self, config, env, policy_model, value_model=None):
        self.config = config
        self.env = env
        
        # Ensure models are on the correct device
        # Get device from tsilva_notebook_utils if available, otherwise infer from model
        try:
            from tsilva_notebook_utils.torch import get_default_device
            device = get_default_device()
        except ImportError:
            # For shared backbone models, get device from the shared model
            if hasattr(policy_model, 'shared_net'):
                device = next(policy_model.shared_net.parameters()).device
            else:
                device = next(policy_model.parameters()).device
        
        # Check if these are wrapper models that share a backbone
        if hasattr(policy_model, 'shared_net') and hasattr(value_model, 'shared_net'):
            # For shared backbone models, ensure the shared model is on the correct device
            policy_model.shared_net.to(device)
            # Force eval mode for rollout collection
            policy_model.shared_net.eval()
            self.policy_model = policy_model
            self.value_model = value_model
        else:
            # For separate models, move each to device
            self.policy_model = policy_model.to(device)
            self.value_model = value_model.to(device) if value_model is not None else None
            # Force eval mode for rollout collection
            self.policy_model.eval()
            if self.value_model:
                self.value_model.eval()
        self.last_obs = None
        
    def start(self):
        """Start the collector"""
        pass
        
    def stop(self):
        """Stop the collector"""
        pass
        
    def update_models(self, policy_state_dict, value_state_dict):
        """Update model weights"""
        pass
        
    def get_rollout(self, timeout=1.0):
        """Get next rollout data"""
        raise NotImplementedError

class SyncRolloutCollector(BaseRolloutCollector):
    """Synchronous rollout collector - collects data on demand"""
    def __init__(self, config, env, policy_model, value_model=None):
        super().__init__(config, env, policy_model, value_model=value_model)
        
    def get_rollout(self, timeout=1.0):
        trajectories, extras = collect_rollouts(
            self.env,
            self.policy_model,
            self.value_model,
            n_steps=self.config.train_rollout_steps,
            last_obs=self.last_obs
        )
        
        self.last_obs = extras['last_obs']
        return trajectories

class AsyncRolloutCollector(BaseRolloutCollector):
    """Background thread that continuously collects rollouts using latest model weights"""
    def __init__(self, config, env, policy_model, value_model=None):
        super().__init__(config, env, policy_model, value_model)
        
        # Thread-safe queue for rollout data
        self.rollout_queue = queue.Queue(maxsize=3)  # Buffer 3 rollouts max
        
        # Store the target device from the original models
        if hasattr(policy_model, 'shared_net'):
            self.device = next(policy_model.shared_net.parameters()).device
        else:
            self.device = next(policy_model.parameters()).device
        
        # Shared model weights (CPU copies for thread safety)
        self.policy_state_dict = None
        self.value_state_dict = None
        self.model_lock = threading.Lock()
        
        # Control flags
        self.running = False
        self.thread = None

        self.weights_version = 0  # Version counter for model updates
        self.model_version = 0  # Version counter for model updates
        
    def start(self):
        """Start the background rollout collection thread"""
        if self.running:
            return
        
        # Initialize with current model weights before starting thread
        if hasattr(self.policy_model, 'shared_net'):
            # For shared backbone, use the shared model's state dict
            initial_state_dict = self.policy_model.shared_net.state_dict()
            self.policy_state_dict = {k: v.cpu().clone() for k, v in initial_state_dict.items()}
            self.value_state_dict = self.policy_state_dict  # Same for both
        else:
            # For separate models, use each model's state dict
            self.policy_state_dict = {k: v.cpu().clone() for k, v in self.policy_model.state_dict().items()}
            if self.value_model:
                self.value_state_dict = {k: v.cpu().clone() for k, v in self.value_model.state_dict().items()}
        
        self.weights_version = 1
        self.model_version = 0  # Force initial update
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the background rollout collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def update_models(self, policy_state_dict, value_state_dict=None):
        """Update model weights from main training thread"""
        with self.model_lock:
            self.policy_state_dict = copy.deepcopy(policy_state_dict)
            if value_state_dict: self.value_state_dict = copy.deepcopy(value_state_dict)
            self.weights_version += 1
            
    def get_rollout(self, timeout=1.0):
        """Get next rollout data (non-blocking with timeout)"""
        try:
            return self.rollout_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # TODO: only load when it changes?
    def _update_model_weights(self):
        if self.model_version >= self.weights_version: return

        """Update local model weights from shared state dicts"""
        with self.model_lock:
            # Check if state dicts are available
            if self.policy_state_dict is None:
                return  # Skip update if no state dict available yet
                
            # Check if these are shared backbone models
            if (hasattr(self.policy_model, 'shared_net') and hasattr(self.value_model, 'shared_net') 
                and self.policy_model.shared_net is self.value_model.shared_net):
                # For shared backbone, update the underlying shared model
                # Move state dict tensors to target device before loading
                device_mapped_state_dict = {k: v.to(self.device) for k, v in self.policy_state_dict.items()}
                self.policy_model.shared_net.load_state_dict(device_mapped_state_dict)
                # Also ensure it's in eval mode for rollout collection
                self.policy_model.shared_net.eval()
            else:
                # For separate models, update each individually
                # Move state dict tensors to target device before loading
                device_mapped_policy_state_dict = {k: v.to(self.device) for k, v in self.policy_state_dict.items()}
                self.policy_model.load_state_dict(device_mapped_policy_state_dict)
                self.policy_model.eval()
                
                if self.value_model and self.value_state_dict: 
                    device_mapped_value_state_dict = {k: v.to(self.device) for k, v in self.value_state_dict.items()}
                    self.value_model.load_state_dict(device_mapped_value_state_dict)
                    self.value_model.eval()
                
            self.model_version = self.weights_version
                
    def _collect_loop(self):        
        while self.running:
            # Update to latest model weights
            self._update_model_weights()
            
            # Collect rollout
            trajectories, extras = collect_rollouts(
                self.env,
                self.policy_model,
                self.value_model,
                n_steps=self.config.train_rollout_steps,
                last_obs=self.last_obs
            )
            
            self.last_obs = extras['last_obs']
            
            # Put rollout in queue (non-blocking, drop if full)
            try:
                self.rollout_queue.put(trajectories, block=False)
            except queue.Full:
                # Queue is full, drop oldest and add new
                try:
                    self.rollout_queue.get_nowait()
                    self.rollout_queue.put(trajectories, block=False)
                except queue.Empty:
                    pass