import tensorflow as tf
import numpy as np
import random
from Games import *

game = TicTacToe()
action_space = game.action_space


def random_move():
    global game
    action = np.random.choice(game.action_space)
    while not game.is_valid(action):
        action = np.random.choice(game.action_space)
    return action


def epsilon_greedy():

    epsilon = 0.1
    if random.random() < 0.1:
        action = random_move()
    else:
        # FIXME: should choose the best choice through the MCTS
        action = random_move()
    return action


n_episodes = 100
for i in range(n_episodes):

    # restart the game
    game.restart()

    while not game.terminal:

        game.render()

        # collect observations
        action = epsilon_greedy()
        reward = game.step(action)

        # now will be the turn of the other player
        game.invert_board()
