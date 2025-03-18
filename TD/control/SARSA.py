from collections import defaultdict

env = gym.make("CartPole-v1")
# n_obs = env.observation_space.n
n_actions = env.action_space.n
Q_values = defaultdict(float)

gamma = 0.99
alpha = 0.1
episodes = 20_000

total_rewards = []

# discritization of states values

def disc_state(obs, bins_per_dimension=30):
    """
    Discretize continuous observation into discrete bins.
    """
    bins = [np.linspace(-x, x, bins_per_dimension + 1)[1: -1]
            for x in [4.8, 10, 0.418, 10]]
    discrete_state = tuple(np.digitize(o, b) for o, b in zip(obs, bins))
    return discrete_state


#exploaration policy
def exp_policy(state, epsilon=1.0):
  if np.random.rand() < epsilon :
    # print("rand")
    return env.action_space.sample()
  else:
    # print("arg")
    return np.argmax([Q_values[(state, a)] for a in range(env.action_space.n)])
# # # update policy
# def max_q(next_state):
#   return np.max([Q_values[(next_state, a)] for a in range(env.action_space.n)])


for episode in range(episodes):
  obs, _ = env.reset()
  state = disc_state(obs)
  done = False
  ep_reward = 0
  epsilon = max(0.01, 1.0 - episode / episodes)
  while not done:
    #explore
    action = exp_policy(state, epsilon)
    next_obs, reward, term, trunc, info = env.step(action)
    #update
    next_state = disc_state(next_obs)
    Q_values[(state, action)] += alpha * (
            reward + gamma * Q_values[(next_state, exp_policy(next_state, epsilon))]
            - Q_values[(state, action)]
    )
    obs = next_obs
    state = disc_state(obs)
    ep_reward += reward
    done = term or trunc
  total_rewards.append(ep_reward)
  if (episode + 1) % 1000 == 0:
    print(f"Episode {episode + 1}:\tReturn = {ep_reward}")