import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

from utils.plot_rewards import plot

class Reinforce:
  def __init__(self, n_states, n_actions, env="CartPole-v1", lr=0.001, gamma=0.99, episodes=2500, timesteps=1000): # for discrete actions
    self.n_states = n_states
    self.n_actions = n_actions
    self.lr = lr
    self.episodes = episodes
    self.timesteps = timesteps

    self.buffer = deque(maxlen=self.timesteps)
    self.gamma = gamma
    self.env = gym.make(env)
    self.model = self.build_model()

  def build_model(self):

      fc1 = nn.Linear(self.n_states, 64)
      fc2 = nn.Linear(64, 64)
      fc3 = nn.Linear(64, self.n_actions)
      relu = nn.ReLU()
      softmax = nn.Softmax(dim=1)
      model = nn.Sequential(fc1, relu, fc2, relu, fc3, softmax)
      return model

  def policy(self, state):
      state = torch.from_numpy(state).float().unsqueeze(0)
      act_probs = self.model(state)
      act_probs = np.array(act_probs.detach())[0]
      action = np.random.choice(self.n_actions, p=act_probs)
      return action

  def push(self, state, action, reward, next_state, done):
      self.buffer.append((state, action, reward, next_state, done))
  def compute_returns(self, rewards):

      returns = []
      G_t = 0
      for reward in reversed(rewards):
          G_t = reward + G_t * self.gamma
          returns.insert(0, G_t)
      return returns

  def train(self):
      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      total_rewards = []
      for episode in range(self.episodes):
          state, _ = env.reset()
          rewards = []
          # reset after each episode
          self.buffer.clear()
          for timestep in range(self.timesteps):
              action = self.policy(state)
              next_state, reward, term, trunc, _ = env.step(action)
              self.push(state, action, reward, next_state, term or trunc)
              state = next_state
              rewards.append(reward)

              if term or trunc:
                  break
          total_rewards.append(rewards)
          returns = self.compute_returns(rewards)
          policy_loss = []
          for (state, action, reward, next_state, done), G_t in zip(self.buffer, returns):
              state = torch.from_numpy(state).float().unsqueeze(0)
              act_probs = self.model(state)
              log_prob = torch.log(act_probs[0, action])
              policy_loss.append(-log_prob * G_t)
          optimizer.zero_grad()
          policy_loss = torch.stack(policy_loss).sum() # sum all losses
          policy_loss.backward()
          optimizer.step()
          if (episode + 1) % 50==0 and episode +1 <= 1000:
             print(f"Episode: {episode+1}\t Reward: {np.sum(rewards)}")
          elif  (episode + 1) % 500==0 and episode +1 > 1000:
             print(f"Episode: {episode+1}\t Reward: {np.sum(rewards)}")
      return total_rewards




model = Reinforce(n_states=4, n_actions=2)
rewards = model.train()

# sum of rewards for single episode
r = [np.sum(r) for r in rewards]

plot(r, "Reinforce")