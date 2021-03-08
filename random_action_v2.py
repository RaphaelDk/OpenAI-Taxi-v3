import gym
import random

def random_action_v2(env):
    epochs, penalties, rewards = 0, 0, 0

    done = False

    while not done:
        actions = []
        max_reward = -10

        for i in range(6):
            max_reward = max(max_reward, env.P[env.s][i][0][2])

        for i in range(6):
            if (i >= 4 or env.P[env.s][i][0][1] != env.s) and env.P[env.s][i][0][2] == max_reward:
                actions.append(i)

        action = actions[random.randint(0, len(actions) - 1)]

        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]