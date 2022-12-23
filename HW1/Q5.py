import math
import matplotlib.pyplot as plt
import random
import numpy as np

carrier_arr = []

for _ in range(2000):
    N = 1000                         # N = number of rounds
    K = 4                            # K = number of arms
    alpha = 2                        # alpha = exploration factor (0.5 <= aplha <= 2)
    X = [0, 0, 0, 0]                 # X = array of pulls
    T = [0, 0, 0, 0]                 # T = array of selections
    Mean = [0.5, 0.8, 0.3, 0.4]      
    optMean = max(Mean)

    arm_sel = []                     # Array to T after each round
    reward_arr = []

    for n in range(1, N+1):
        maxQ = 0
        index_max_Q = 0
        
        if n <= K:                   # first we pull each arm for initialization
            T[n-1] = T[n-1] + 1      # updating selections of arms. an = n-1 as loop is from [1, N]
            val = random.random()    # calculating a random float value in [0, 1]
            if val <= Mean[n-1]:     # assigning reward
                reward = 1           # 1 if mean of that arm >= random float else reward = 0
            else:
                reward = 0
            X[n-1] = X[n-1] + reward # updating pulls according to reward assigned
            reward_arr.append(reward)

            arm_sel.append(n-1)

        else:
            for k in range(K):       # after initializing, we find the Q for each round
                observedAverage = X[k]/T[k]
                confidenceInterval = math.sqrt((alpha*math.log(n))/T[k])
                Q = observedAverage + confidenceInterval   # Q = Confidence Bound

                if Q > maxQ:         # calculating max Q for each round
                    maxQ = Q
                    index_max_Q = k  # an = index of max(Q)

            T[index_max_Q] = T[index_max_Q] + 1            # updating selections of arms. an = index of max(Q)
            val = random.random()    # calculating a random float value in [0, 1] and assigning reward as above
            if val <= Mean[index_max_Q]:
                reward = 1
            else:
                reward = 0
            X[index_max_Q] = X[index_max_Q] + reward       # updating pulls according to reward assigned
            reward_arr.append(reward)

            arm_sel.append(index_max_Q)

    avgReward = np.average(np.array(reward_arr))
    carrier_arr.append(avgReward)

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

arr1 = Q2(0.1)

plt.figure()
plt.plot(arr1)
plt.plot(carrier_arr)

plt.show()