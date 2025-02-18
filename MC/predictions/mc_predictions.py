# !pip install swig

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


env = gym.make("FrozenLake-v1", render_mode="rgb_array")
trigger = lambda t:t%50_000==0
env = RecordVideo(env = env, video_folder="./episodes_videos", episode_trigger=trigger)

num_states = env.observation_space.n
num_actions = env.action_space.n


total_returns = defaultdict(float)
N = defaultdict(int)
def random_policy(state):
  return env.action_space.sample()

def generate_episode(env):
    num_timesteps = 100
    episode = []
    for t in range(num_timesteps):
      state, info = env.reset()
      done = False
      while not done:
        action = random_policy(env)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated

    return episode

def update_returns(episode):
        M = 1000
        for i in range(M):
            if i+1 % 100 == 0:
                print(f"Episode {i} of {M}")
            episode = generate_episode(env)

            states, actions, rewards = zip(*episode)

            for t, state in enumerate(states):
                R = sum(rewards[t:])
                total_returns[state] += R
                N[state] +=1
##################################edit########################
import pandas as pd

total_return_df = pd.DataFrame(total_returns.items(), columns=['state', 'total_return'])
N_df= pd.DataFrame(N.items(), columns=["state", "N"])
"""
total_returns = defaultdict(float)
N = defaultdict(int)
"""
for state in range(num_states):
    if state not in total_returns:
        total_returns[state] = 0.0  # Assign a default value for unvisited states
    if state not in N:
        N[state] = 0  # Assign a default visit count for unvisited states

# Create DataFrames
total_return_df = pd.DataFrame(total_returns.items(), columns=['state', 'total_return'])
N_df = pd.DataFrame(N.items(), columns=["state", "N"])

# Merge the two DataFrames on 'state' to get total returns and visit counts
df = pd.merge(total_return_df, N_df, on="state", how="right")  # 'right' join to include all states
df = df.sort_values(by="state")  # Sort by 'state' for clarity

# Display the first 15 rows of the final DataFrame
df.head(15)

df["V"] = df["total_return"]/df["N"]
df.head(15)