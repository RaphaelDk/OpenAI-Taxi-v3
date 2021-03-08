import sys
import gym
import random
import numpy as np

def q_learning(env):
    with open('q_learning.npy', 'rb') as f:
        q_table = np.load(f)

    epochs, penalties, rewards = 0, 0, 0

    done = False
    state = env.s

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]

def train():
    env = gym.make('Taxi-v3').env

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.2512238484351891
    gamma = 0.7749915552696941
    epsilon = 0.9957089031634627
    epsilon_decay = 0.8888782926665223

    for i_episode in range(5000):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            state = next_state

        epsilon = max(epsilon * epsilon_decay, 0.1)

    with open('q_learning.npy', 'wb') as f:
        np.save(f, q_table)
    

if __name__ == "__main__":
    train()
