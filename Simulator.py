import random
import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, n, params):
        self.n = n # number of bandits
        self.params = params # probability parameters of bandits

    def simulate(self, algorithm, play_rounds, repeat_times):
        regret = np.zeros(play_rounds+1)
        max_reward = np.array(range(play_rounds+1)) * max(self.params)
        for repeat_idx in range(repeat_times):
            reward = 0
            algorithm.reset()
            for t in range(1, play_rounds+1):
                i = algorithm.select(t)
                param = self.params[i]
                sample = random.uniform(0, 1)
                if param >= sample:
                    r = 1
                else:
                    r = 0
                algorithm.receive(i, r)
                reward += r
                regret[t] += max_reward[t] - reward
        regret = regret / repeat_times
        return regret


class NaiveAlgorithm:
    def __init__(self, n, explore_rounds):
        self.n = n
        self.explore_rounds = explore_rounds
        self.sum_reward = None
        self.fixed_selection = None
        self.idx_max_param = None
        
    def reset(self):
        self.sum_reward = np.zeros(self.n)
        self.fixed_selection = np.zeros(self.n * self.explore_rounds + 1, dtype=int)
        t = 1
        for i in range(self.n):
            for j in range(self.explore_rounds):
                self.fixed_selection[t] = i
                t += 1

    def select(self, t):
        if t <= self.n * self.explore_rounds:
            return self.fixed_selection[t]
        else: 
            if t == self.n * self.explore_rounds + 1:
                self.idx_max_param = np.argmax(self.sum_reward)
            return self.idx_max_param
            
    def receive(self, i, r):
        self.sum_reward[i] += r


if __name__ == '__main__':
    params = [0.2, 0.3]
    n = len(params)
    simulator = Simulator(n, params)
    algorithm = NaiveAlgorithm(n, 100)
    regret = simulator.simulate(algorithm, 10000, 100)
    plt.plot(regret)
    plt.xscale('log')
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.show()

