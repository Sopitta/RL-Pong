import matplotlib.pyplot as plt
import numpy as np

avg_episode_rewards =np.load('avg_episode_rewards_st_1.npy')
episode_rewards = np.load('episode_rewards_st_1.npy')
avg_episode_lengths = np.load('avg_episode_lengths_st_1.npy')
episode_lengths = np.load('episode_lengths_st_1.npy')

plt.plot(episode_rewards)
plt.plot(avg_episode_rewards)
plt.legend(["Mean reward", "100-episode average"])
plt.savefig("PPO_rewards_4.png")

plt.close()
plt.ioff()
plt.plot(episode_lengths)
plt.plot(avg_episode_lengths)
plt.legend(["Mean episode lengths", "100-episode average"])
plt.savefig("PPO_ep_lengths_4.png")
