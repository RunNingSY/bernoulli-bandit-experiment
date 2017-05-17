# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:45:30 2017

@author: RunNing
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import abc


class Algorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self):
        return

    @abc.abstractmethod
    def select(self, t):
        return

    @abc.abstractmethod
    def receive(self, i, r):
        return
        
        
class Simulator:
    def __init__(self, n, params):
        self.n = n
        self.params = params

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


class EpsilonGreedy:
    def __init__(self, n, epsilon):
        self.n = n
        self.epsilon = epsilon
        self.sum_reward = None
        self.mean_reward = None
        self.counts = None

    def reset(self):
        self.sum_reward = np.zeros(self.n)
        self.mean_reward = np.zeros(self.n)
        self.counts = np.zeros(self.n)

    def select(self, t):
        sample = random.uniform(0, 1)
        if sample < self.epsilon:
            flag = True
        else:
            flag = False
        if flag is True:
            i = random.randint(0, self.n-1)
        else:
            i = np.argmax(self.mean_reward)
        return i

    def receive(self, i, r):
        self.counts[i] += 1
        self.sum_reward[i] += r
        self.mean_reward[i] = self.sum_reward[i] / self.counts[i]


class UCB:
    def __init__(self, n):
        self.n = n
        self.fixed_selection = None
        self.sum_reward = None
        self.mean_reward = None
        self.counts = None

    def reset(self):
        self.sum_reward = np.zeros(self.n)
        self.mean_reward = np.zeros(self.n)
        self.counts = np.zeros(self.n)
        self.fixed_selection = np.arange(-1, self.n)

    def select(self, t):
        if t <= self.n:
            return self.fixed_selection[t]
        else:
            upper_confidence_bounds = self.mean_reward + np.sqrt(2*np.log(t)/self.counts)
            i = np.argmax(upper_confidence_bounds)
            return i

    def receive(self, i, r):
        self.counts[i] += 1
        self.sum_reward[i] += r
        self.mean_reward[i] = self.sum_reward[i] / self.counts[i]


class TS:
    def __init__(self, n):
        self.n = n
        self.beta_S = None
        self.beta_F = None

    def reset(self):
        self.beta_S = np.ones(self.n)
        self.beta_F = np.ones(self.n)

    def select(self, t):
        theta = np.zeros(self.n)
        for i in range(self.n):
            theta[i] = random.betavariate(self.beta_S[i], self.beta_F[i])
        return np.argmax(theta)

    def receive(self, i, r):
        if r == 1:
            self.beta_S[i] += 1
        else:
            self.beta_F[i] += 1


Algorithm.register(NaiveAlgorithm)
Algorithm.register(EpsilonGreedy)
Algorithm.register(UCB)
Algorithm.register(TS)


def cpu_lower_bound(params, play_rounds):
        def kl_divergence(p, q):
            if p == q:
                return 1e-9
            else:
                return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
        p_max = max(params)
        coef = np.sum([(p_max-p)/kl_divergence(p, p_max) for p in params])
        regret = np.arange(play_rounds+1, dtype=float)
        regret[1:] = coef*np.log(regret[1:])
        return regret
        
        
def run_experiment(setting):
    play_rounds = 100000
    repeat_times = 100
    n = setting[0]
    notation = setting[1]
    params = setting[2]
    simulator = Simulator(n, params)
    lower_bound = cpu_lower_bound(params, play_rounds)
    algorithms = [[NaiveAlgorithm(n, 100), 'Naive'],
                  [EpsilonGreedy(n, 0.1), 'Greedy'],
                  [UCB(n), 'UCB'],
                  [TS(n), 'TS']]
    result = [[simulator.simulate(algorithm, play_rounds, repeat_times), name]
              for algorithm, name in algorithms]
                
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    for regret, name in result:
        plt.plot(regret, label=name)
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.title(notation+' nomal axis')
    plt.subplot(122)
    for regret, name in result:
        plt.plot(regret, label=name)
    plt.plot(lower_bound, label='lower_bound')
    plt.legend(loc='upper left', frameon=False)
    plt.xscale('log')
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.title(notation+' log axis')
    plt.savefig(notation+'.png', dpi=600)
    
    
if __name__ == '__main__':
    from multiprocessing import Pool
    settings = [[2, 'A1', [0.9, 0.8]],
                [2, 'A2', [0.6, 0.5]],
                [2, 'A3', [0.9, 0.2]]]
    settings = [[2, 'A1', [0.9, 0.8]],
                [2, 'A2', [0.6, 0.5]],
                [2, 'A3', [0.9, 0.2]],
                [5, 'B1', [0.9, 0.88, 0.86, 0.84, 0.82]],
                [5, 'B2', [0.6, 0.58, 0.56, 0.54, 0.52]],
                [5, 'B3', [0.9, 0.7, 0.5, 0.3, 0.1]],
                [15, 'C1', [0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78,
                            0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.64, 0.62]],
                [15, 'C2', [0.65, 0.63, 0.61, 0.59, 0.57, 0.55, 0.53,
                            0.51, 0.49, 0.47, 0.45, 0.43, 0.41, 0.39, 0.37]],
                [15, 'C3', [0.89, 0.83, 0.77, 0.71, 0.65, 0.59, 0.53,
                            0.47, 0.41, 0.35, 0.29, 0.23, 0.17, 0.11, 0.05]]]
                            
    pool = Pool()
    pool.map(run_experiment, settings)
    pool.close()
    pool.join()
                            

        
    
        