
import numpy as np 
import matplotlib.pyplot as plt
def plot(reward, alg_name,  window_size = 50):
    """
    Total return per episode ...
    """
    avg_reward = [np.mean(reward[max(0, i - window_size): (i+1)]) for i in range(len(reward))]
    std_reward = [np.std(reward[max(0, i - window_size): (i+1)]) for i in range(len(reward))]

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward, label="Total Return", color="blue")

    # area between mean + std & mean - std
    plt.fill_between(
        range(len(std_reward)),
        np.array(avg_reward) - np.array(std_reward),
        np.array(avg_reward) + np.array(std_reward),
        color="blue", alpha=0.2, label="Std"
    )

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{alg_name} Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    