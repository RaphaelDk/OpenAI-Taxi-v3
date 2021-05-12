import sys
import gym
import random
import numpy as np

def q_learning(env):
    # Charger la q_table
    with open('q_learning.npy', 'rb') as f:
        q_table = np.load(f)

    epochs, penalties, rewards = 0, 0, 0

    done = False
    # Récupérer l'état de la partie
    state = env.s

    while not done:
        # Choisir une action en fonction de l'état actuel
        action = np.argmax(q_table[state])
        # Récupérer l'état et autres info en fonction de l'action
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]

def train():
    env = gym.make('Taxi-v3').env

    # Créer une table remplie de zéros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # alpha / beta -> paramètres formule mathématique
    alpha = 0.2512238484351891
    gamma = 0.7749915552696941
    # Plus l'epsilon est elevé, plus le taxi explorera au hasard
    epsilon = 0.9957089031634627
    # Epsilon est réduit de ce nombre à chaque itération
    epsilon_decay = 0.8888782926665223

    # 5000 itérations pour entrainer (bon ration temps / perf sans overfitting)
    for i_episode in range(5000):
        state = env.reset()
        done = False

        while not done:
            # Lors des premières itérations le taxi est "greedy"
            # puis se base sur ce qu'il a appris au fur et à mesure
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)

            # Récupérer la valeur actuelle de la q_table
            actual_value = q_table[state, action]
            # La valeur maximale en fonction du state du prochain "coup" est utilisé dans la formule en dessous
            next_max = np.max(q_table[next_state])
            # Calcul de la nouvelle valeur
            new_value = (1 - alpha) * actual_value + alpha * (reward + gamma * next_max)
            # Mettre à jour la q_table
            q_table[state, action] = new_value

            state = next_state

        # Décrémentation d'epsilon avec un minimum à 0.1
        epsilon = max(epsilon * epsilon_decay, 0.1)

    with open('q_learning.npy', 'wb') as f:
        np.save(f, q_table)
    

if __name__ == "__main__":
    train()
