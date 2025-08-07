import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")# gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")#, render_mode="human")
obs, _ = env.reset()
print(obs)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

episode = 0
episode_reward = 0
exploration_prate = 0.99  # epsilon-greedy exploration rate
for _ in range(1000000):
    # get argmax from q table
    if np.random.rand() < exploration_rate:  # epsilon-greedy action selection
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[obs])

    action = action.item()
    next_obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    q_table[obs, action] = reward + 0.99 * np.max(q_table[next_obs])
    obs = next_obs
    #env.render()
    if terminated or truncated:
        obs, info = env.reset()
        print(f"Episode {episode} finished with reward {episode_reward}")
        episode += 1
        episode_reward = 0
        exploration_rate *= 0.99999  # decay exploration rate
        print(exploration_rate)

env = gym.make("Taxi-v3", render_mode="human")
obs, _ = env.reset()

episode = 0
episode_reward = 0
for _ in range(1000000):
    action = np.argmax(q_table[obs])
    action = action.item()
    next_obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    obs = next_obs
    if terminated or truncated:
        obs, info = env.reset()
        print(f"Episode {episode} finished with reward {episode_reward}")
        episode += 1
        episode_reward = 0