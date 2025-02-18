import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env.reset()

def value_fn(env, gamma=1.0):

  n_states = env.observation_space.n
  n_actions = env.action_space.n
  env = env.unwrapped # To use the transition probablity method env.P[s][a]
  value_table = np.zeros(n_states)
  num_iterations = 1000

  threshold = 1e-4 # To stop the iteration if the the values converged

  for i in range(num_iterations):
    new_value_table = value_table.copy()
    for s in range(n_states):
      Q_values = []
      for a in range(n_actions):
        q_value = 0
        for Pr, s_, r, done in env.P[s][a]:
          q_value += Pr * (r + gamma * new_value_table[s_])
        Q_values.append(q_value)
      value_table[s] = max(Q_values)
    if np.sum(np.fabs( value_table - new_value_table)) <= threshold:
      print(f"Converged after {i+1} iterations")
      break
  return value_table

# Optimal value table
optimal_value_table = value_fn(env, 1.0)


def extract_policy(env, optimal_value_table, gamma=1.0):
  n_states = env.observation_space.n
  n_actions = env.action_space.n
  env = env.unwrapped # To use the transition probablity method env.P[s][a]

  policy = np.zeros(n_states)
  for s in range(n_states):
      Q_values = []
      for a in range(n_actions):
        q_value = 0
        for Pr, s_, r, done in env.P[s][a]:
          q_value += Pr * (r + gamma * optimal_value_table[s_])
        Q_values.append(q_value)
      policy[s] = np.argmax(Q_values)
  return policy

# Optimal policy
optimal_policy = extract_policy(env, optimal_value_table, 1.0)


def rollout_w_optimal_policy(env, opitmal_policy, num_episodes = 500, max_timesteps = 50, gamma=1.0):
  episodic_returns = []
  trigger = lambda t:t%1000==0
  env = RecordVideo(env, episode_trigger=trigger, video_folder="./Value_iteration")
  for i in range(num_episodes):
      s, _ = env.reset()
      returns = 0
      for t in range(max_timesteps):
          a = optimal_policy[int(s)]
          s_, r, term, trunc, info = env.step(int(a))
          returns += r
          if term or trunc:
            break
          s = s_
      if i % 100==0:
        print(f"Epsiode ({i+1}/{num_episodes}): Returns: {returns} for episode :{i+1}")
      episodic_returns.append(returns)
  return episodic_returns


def plot_performance(episodic_returns, chunk_size=50):
    # Average return per chunk
    avg_returns = [np.mean(episodic_returns[i:i+chunk_size])
                   for i in range(0, len(episodic_returns), chunk_size)]

    std_returns = [np.std(episodic_returns[i:i + chunk_size]) for i in range(0, len(episodic_returns), chunk_size)]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(avg_returns))
    plt.plot(x, avg_returns, label="Average Return")
    plt.fill_between(x, np.array(avg_returns) - np.array(std_returns), np.array(avg_returns) + np.array(std_returns), color='b', alpha=0.2, label="Variance")
    plt.xlabel(f"Chunk ({chunck_size} Episodes)")
    plt.ylabel("Average Return")
    plt.title("Performance Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

episodic_returns = rollout_w_optimal_policy(env, optimal_policy, num_episodes = 10_000, gamma=1.0)

# In the rollout, the env should be set to is_slippery=False
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
plot_performance(episodic_returns)