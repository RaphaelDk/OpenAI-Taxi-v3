import gym

def random_action(env):
    epochs, penalties, rewards = 0, 0, 0

    done = False

    while not done:
        # Récupérer les actions possibles et en choisir une au hasard
        action = env.action_space.sample()
        # Récupérer l'état et autres info en fonction de l'action
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        epochs += 1

    return [epochs, rewards / epochs, penalties]