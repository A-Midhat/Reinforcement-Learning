import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env.reset()

def compute_value_table(env, policy, gamma=1.0):

  n_states = env.observation_space.n
  n_actions = env.action_space.n
  env = env.unwrapped # To use the transition probablity method env.P[s][a]
  value_table = np.zeros(n_states)
  num_iterations = 100
  threshold = 1e-10

  for i in range(num_iterations):
    new_value_table = value_table.copy()
    for s in range(n_states):
      a = policy[s] # random policy
      v_value = 0
      for Pr, s_, r, done in env.P[s][a]:
        v_value += Pr * (r + gamma * new_value_table[s_])


      value_table[s] = v_value
    if np.sum(np.abs(new_value_table - value_table)) < threshold:
        break


  return value_table

def extract_policy_func(env, value_table, gamma=1.0):
  n_states = env.observation_space.n
  n_actions = env.action_space.n
  env = env.unwrapped # To use the transition probablity method env.P[s][a]

  policy = np.zeros(n_states)
  for s in range(n_states):
      Q_values = []
      for a in range(n_actions):
        q_value = 0
        for Pr, s_, r, done in env.P[s][a]:
          q_value += Pr * (r + gamma * value_table[s_])
        Q_values.append(q_value)
      policy[s] = np.argmax(Q_values)
  return policy

def policy_iteration_func(env, num_iterations = 1000):
    # generate random policy
    policy = [env.action_space.sample() for i in range(env.observation_space.n)]
    for i in range(num_iterations):
      value_table_ = compute_value_table(env, policy)
      new_policy = extract_policy_func(env, value_table_)
      # if np.all(policy == new_policy):
      if np.array_equal(policy, new_policy):
        print(f"Converged after {i+1} iterations")
        break
      policy = new_policy
    return value_table_, policy

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
      if (i+1) % 100==0:
        print(f"Epsiode ({i+1}/{num_episodes}):\t Returns: {returns}")
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


optimal_value_table, optimal_policy = policy_iteration_func(env, 1000)

env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
episodic_returns = rollout_w_optimal_policy(env, optimal_policy, num_episodes = 10_000, gamma=1.0)
plot_performance(episodic_returns)