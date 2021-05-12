import gym
import random

def random_action_v2(env):
    epochs, penalties, rewards = 0, 0, 0

    done = False

    while not done:
        actions = []
        max_reward = -10

        # Récupérer la récompense maximale
        for i in range(6):
            max_reward = max(max_reward, env.P[env.s][i][0][2])

        # Stocker les actions menant à la récompense maximale
        for i in range(6):
            # Si l'action entraine la récompense maximale ET
            # qu'elle permette de se déplacer (ne pas rentrer dans un mur) ou récupérer/déposer un passager
            if env.P[env.s][i][0][2] == max_reward and (i >= 4 or env.P[env.s][i][0][1] != env.s):
                actions.append(i)

        # Choisir une des actions donnant la récompense maximale au hasard
        action = actions[random.randint(0, len(actions) - 1)]

        # Récupérer l'état et autres info en fonction de l'action
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]