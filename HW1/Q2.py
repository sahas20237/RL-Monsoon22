import numpy as np
import matplotlib.pyplot as plt

def Q2(eps):
    action_space = [1, 2, 3, 4]
    T = 1000

    Q_value = [0, 0, 0, 0]
    pull_count = [0, 0, 0, 0]
    rewards = []

    for i in range(1, T+1):
        x = np.random.random()
        if x < eps:
            index = np.random.choice(len(action_space))
        else:
            index = np.argmax(Q_value)

        if index == 0:
            reward = np.random.normal(0, 1)
        if index == 1:
            reward = np.random.normal(0, 0.7)
        if index == 2:
            reward = np.random.normal(0, 0.2)
        if index == 3:
            reward = np.random.normal(0.2, 0.8)

        pull_count[index] += 1
        Q_value[index] = Q_value[index] + (1/pull_count[index] * reward)

        rewards.append(reward)
    
    return rewards

# (i)
arr1 = Q2(0.2)
print(np.average(arr1))
plt.figure()
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('e = 0.2')
plt.plot(arr1)

# (ii)
arr2 = Q2(0.8)
print(np.average(arr2))
plt.figure()
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('e = 0.8')
plt.plot(arr2)

# (iii)
arr3 = Q2(0)
print(np.average(arr3))
plt.figure()
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('e = 0')
plt.plot(arr3)

# (iv)
arr4 = Q2(1)
print(np.average(arr4))
plt.figure()
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('e = 1')
plt.plot(arr4)

plt.show()