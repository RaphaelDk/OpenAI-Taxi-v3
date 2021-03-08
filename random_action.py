import gym

def random_action(env):
    epochs, penalties, rewards = 0, 0, 0

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]