"""
!pip install  gymnasium
!pip install swig wheels
!pip install gymnasium[box2d]
"""

import gymnasium as gym
import numpy as np
import pandas as pd
ENV = gym.make("CartPole-v1")

##Normal TD pridiction

# V(s) += alpha * (R + gamma*V(s_) - V(s))
n_obs = ENV.observation_space.shape[0]
# n_obs = ENV.observation_space.n
alpha= 0.85
gamma = 0.95
# value_table = np.zeros(n_obs)
from collections import defaultdict
from tqdm import tqdm


value_table = defaultdict(float)
def policy(state, epsilon=0.1):
  if np.random.rand() < epsilon:
    return ENV.action_space.sample()
  else:
    return np.argmax([value_table[(state, a)] for a in range(ENV.action_space.n)])
def disc_state(obs, bins_per_dimension):
    """
    Discretize continuous observation into discrete bins.
    """
    bins = [np.linspace(-x, x, bins_per_dimension + 1)[1: -1]
            for x in [4.8, 10, 0.418, 10]]
    discrete_state = tuple(np.digitize(o, b) for o, b in zip(obs, bins))
    return discrete_state

episodes_returns = []
for episode in range(25_000):
  obs, info = ENV.reset()
  state = disc_state(obs, 100)
  episodic_return = 0.0
  if episode >=12_000:
    epsilon = 0.01
  else:
    epsilon = 0.1
  done = False
  while not done:
    action = policy(state, epsilon)
    next_obs, r, term, trunc, info = ENV.step(action)
    next_state = disc_state(next_obs, 100)
    value_table[(state, action)] += alpha * (
            r + gamma * max(value_table[(next_state, a)] for a in range(ENV.action_space.n))
            - value_table[(state, action)]
        )
    state = next_state
    episodic_return += r
    done = term or trunc
  # for t in range(500):
  #   action = policy(state, epsilon)
  #   next_obs, r, term, trunc, info = ENV.step(action)
  #   next_state = disc_state(next_obs, 100)
  #   value_table[(state, action)] += alpha * (
  #           r + gamma * max(value_table[(next_state, a)] for a in range(ENV.action_space.n))
  #           - value_table[(state, action)]
  #       )
  #   state = next_state
  #   episodic_return += r
  #   if term or trunc:
  #     break

  episodes_returns.append(episodic_return)

  if (episode + 1) % 1000 == 0:
      print(f"Episode {episode + 1}:\tReturn = {episodic_return}")


import matplotlib.pyplot as plt
window_size = 70
rolling_avg = pd.Series(episodes_returns).rolling(window=window_size).mean()
rolling_std = pd.Series(episodes_returns).rolling(window=window_size).std()

plt.figure(figsize=(10, 6))
plt.plot(rolling_avg, label="Average Return")
plt.fill_between(range(len(episodes_returns)), rolling_avg - rolling_std, rolling_avg + rolling_std, alpha=0.3, label="Error")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.title("Episodic Return")
plt.legend()
plt.grid(True)
plt.show()

def extract_policy(value_table, env, bins_per_dimension):
    """
    Extract the optimal policy from the value table.

    Parameters:
    value_table: dict - Q-values for state-action pairs.
    env: gym.Env - The environment.
    bins_per_dimension: int - Number of bins used for discretization.

    Returns:
    dict - Optimal action for each discretized state.
    """
    policy = {}
    action_space = env.action_space.n  # Number of actions (e.g., 2 for CartPole)

    # Loop through all discretized states
    for state_action, q_value in value_table.items():
        state, action = state_action

        # Check if state exists in the policy table
        if state not in policy:
            policy[state] = action
        else:
            # Update policy if Q-value for this action is better
            if q_value > value_table[(state, policy[state])]:
                policy[state] = action

    return policy
extract_policy(value_table, ENV, 100)

def rollout(env, policy, bins_per_dimension, max_steps=200):
    """
    Run an episode using the given policy.

    Parameters:
    env: gym.Env - The environment.
    policy: dict - The optimal policy.
    bins_per_dimension: int - Number of bins used for discretization.
    max_steps: int - Maximum steps in the episode.

    Returns:
    float - Total reward from the episode.
    """
    obs, _ = env.reset()
    state = disc_state(obs, bins_per_dimension)
    total_reward = 0

    for _ in range(max_steps):
        action = policy.get(state, env.action_space.sample())  # Use policy or random
        obs, reward, term, trunc, _ = env.step(action)
        state = disc_state(obs, bins_per_dimension)
        total_reward += reward

        if term or trunc:
            break

    return total_reward

# Example: Run rollout
optimal_policy = extract_policy(value_table, ENV, bins_per_dimension=100)
for ep in range(100):
    total_reward = rollout(ENV, optimal_policy, bins_per_dimension=100)
    print(f"Episode {ep + 1}: Total Reward = {total_reward}")
print("Total Reward from Rollout:", total_reward)
