import matplotlib.pyplot as plt
import numpy as np
import pickle

# reward_train = pickle.load(open('trainreward', 'rb'))
# reward_dd = pickle.load(open('ddqnrewarddd', 'rb'))
# reward_du = pickle.load(open('duelreward', 'rb'))
# reward_dd_du = pickle.load(open('duelddreward', 'rb'))

reward_train = pickle.load(open('trainloss', 'rb'))
reward_dd = pickle.load(open('ddqnloss', 'rb'))
reward_du = pickle.load(open('duelloss', 'rb'))
reward_dd_du = pickle.load(open('duelddloss', 'rb'))

x_reward_train = np.linspace(1, len(reward_train), len(reward_train))
x_reward_dd = np.linspace(1, len(reward_dd), len(reward_dd))
x_reward_du = np.linspace(1, len(reward_du), len(reward_du))
x_reward_dd_du = np.linspace(1, len(reward_dd_du), len(reward_dd_du))

x_reward_train.astype(int)
x_reward_dd.astype(int)
x_reward_du.astype(int)
x_reward_dd_du.astype(int)

#plt.figure(1)
plt.plot(x_reward_train, reward_train)
plt.title('Reward vs episodes for DQN architecture')
plt.ylabel('Reward')
plt.xlabel('Number of episodes')
#plt.legend([loc='upper left')
plt.show()

#plt.figure(2)
plt.plot(x_reward_dd, reward_dd)
plt.title('Reward vs episodes for DDQN architecture')
plt.ylabel('Reward')
plt.xlabel('Number of episodes')
#plt.legend([loc='upper left')
plt.show()

#plt.figure(3)
plt.plot(x_reward_du, reward_du)
plt.title('Loss vs episodes for Duel-DQN architecture')
plt.ylabel('Loss')
plt.xlabel('Number of episodes')
#plt.legend([loc='upper left')
plt.show()

#plt.figure(4)
plt.plot(x_reward_dd_du, reward_dd_du)
plt.title('Reward vs episodes for Duel-DDQN architecture')
plt.ylabel('Reward')
plt.xlabel('Number of episodes')
#plt.legend([loc='upper left')
plt.show()
