#!/usr/bin/env python3

import sys
import gym
import json
import time
import numpy as np
from random_action import random_action
from random_action_v2 import random_action_v2
from q_learning import q_learning

env = gym.make('Taxi-v3').env
agents = { "random_action": random_action,
           "random_action_v2": random_action_v2,
           "q_learning": q_learning
         }

iterations = int(sys.argv[2])
epochs = []
rewards_per_epoch = []
penalities = []
durations = []

for i_episode in range(iterations):
    env.reset()

    start = time.time()
    stats = agents[sys.argv[1]](env)
    end = time.time()

    epochs.append(stats[0])
    rewards_per_epoch.append(stats[1])
    penalities.append(stats[2])
    durations.append((end - start) * 1000.0)

env.close()

with open('benchmark.json') as json_file:
    benchmarks = json.load(json_file)
    total_iterations = (benchmarks[sys.argv[1]]['iterations'] + iterations)

    benchmarks[sys.argv[1]]['epochs'] = (benchmarks[sys.argv[1]]['epochs'] * benchmarks[sys.argv[1]]['iterations'] + np.sum(epochs)) / total_iterations
    benchmarks[sys.argv[1]]['rewards_per_epoch'] = (benchmarks[sys.argv[1]]['rewards_per_epoch'] * benchmarks[sys.argv[1]]['iterations'] + np.sum(rewards_per_epoch)) / total_iterations
    benchmarks[sys.argv[1]]['penalities'] = (benchmarks[sys.argv[1]]['penalities'] * benchmarks[sys.argv[1]]['iterations'] + np.sum(penalities)) / total_iterations
    benchmarks[sys.argv[1]]['duration'] = (benchmarks[sys.argv[1]]['duration'] * benchmarks[sys.argv[1]]['iterations'] + np.sum(durations)) / total_iterations
    benchmarks[sys.argv[1]]['iterations'] = total_iterations

with open('benchmark.json', 'w') as json_file:
    json.dump(benchmarks, json_file)
