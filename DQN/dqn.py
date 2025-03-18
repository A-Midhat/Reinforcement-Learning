# replay buffer :
"""
1.Initialize the replay buffer DD.
2. For each episode perform step 3.
3. For each step in the episode:
  1. Make a transition, that is, perform an action a in the state s, move to
    the next state ss′, and receive the reward r.
  2. Store the transition information (sss sss sss ss′

)
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
from collections import deque # for better queue and fifo operations
env = gym.make("CartPole-v1")
env.reset()

class ReplayBuffer:
  def __init__(self, capacity=100_000):
    self.capacity = capacity
    self.buffer = deque(maxlen=capacity)
    # self.position = 0

  def push(self, state, action, reward, next_state, done): # add done (term or trunc)

     self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size=32):
    if len(self.buffer) < batch_size:
      raise ValueError("Buffer size is less than batch size")
    return zip(*random.sample(self.buffer, batch_size))

  def __len__(self):
    return len(self.buffer)


class QNetwork(nn.Module):
  def __init__(self, n_state, n_action):
    super(QNetwork, self).__init__()
    self.fc1 = nn.Linear(n_state, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, n_action)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


def policy(net, state, epsilon=0.8):
  if np.random.random() < epsilon:
    return env.action_space.sample()
  else:
    q_values = net(torch.from_numpy(state).float())
    return torch.argmax(q_values).item()
# obs, _ = env.reset()
# policy(main_net, obs), policy(target_net, obs)


# Training loop
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
# exploration net
main_net = QNetwork(n_states, n_actions)
# traget (optimal) net
target_net = QNetwork(n_states, n_actions)
target_net.load_state_dict(main_net.state_dict())
target_net.eval() # for freezing the weights

# Initializing the buffer
D = ReplayBuffer(20_000)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(main_net.parameters()) # lr =0.001

gamma = 0.99
epsilon = 0.2
min_epsilon = 0.01
episodes = 50_000

batch_size = 128
# logs
total_rewards = []
losses = []

# second try
for episode in tqdm.tqdm(range(episodes)):
  state, _ = env.reset()
  ep_reward = 0.0
  ep_loss = 0.0
  done = False
  # after 100k episodes and each interval update the target parameters
  # after 100k start decaying the expolartion rate (epsilon_)

  if episode + 1 > 20_000:
    epsilon = max(min_epsilon, epsilon - episode / episodes)
  if episode + 1 % 500:
    target_net.load_state_dict(main_net.state_dict())
  while not done:

    action = policy(main_net, state, epsilon)
    next_state, reward, term, trunc, _ = env.step(action)
    # Updtae the Buffer
    D.push(state, action, reward, next_state, term or trunc) # I should add the done state, since I will return R if next_State is done

    state = next_state
    ep_reward += reward
    done = term or trunc
    # should try and make it stop doing anything until we store the full buffer

    # if len(D) >= batch_size:
    if episode + 1 > 20_000 and len(D)>= batch_size:
        states, actions, rewards, next_states, dones = D.sample(batch_size)

        states = torch.tensor(states, dtype= torch.float32) # or torch.from_numpy()
        actions = torch.tensor(actions, dtype= torch.int64).unsqueeze(1) # (batch,) ==> (act_batch. 1) or view(-1,1)
        rewards = torch.tensor(rewards, dtype= torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype= torch.float32)
        dones = torch.tensor(dones, dtype= torch.bool).unsqueeze(1)

        with torch.no_grad():
          next_q = target_net(states).gather(1, actions) # gather(dim, idx) //Selects values from the tensor along a specified dimension using the given indices.
          opt_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
          targets = rewards + gamma * (1 - dones.type(torch.float32)) * opt_next_q

        q_values = main_net(states).gather(1, actions)
        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward() # gradients bp
        optimizer.step() # update
        ep_loss += loss.item()
        # logging
  losses.append(ep_loss)

  total_rewards.append(ep_reward)



window_size = 200 # defines how smooth the line becomes

means = [np.mean(total_rewards[max(0, i - window_size): (i + 1)]) for i in range(len(total_rewards))]
stds = [np.std(total_rewards[max(0, i - window_size): (i + 1)]) for i in range(len(total_rewards))]

plt.figure(figsize=(10, 6))
plt.plot(means, label="Average Return", color="blue")
# area between mean + std & mean - std
plt.fill_between(
    range(len(stds)),
    np.array(means) - np.array(stds),
    np.array(means) + np.array(stds),
    color="blue", alpha=0.2, label="Std"
)

plt.xlabel("Epsidoes")
plt.ylabel("Reward")
plt.title("DQN Training")
plt.legend()
plt.grid(True)
plt.show()