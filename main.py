import tensorflow as tf
import numpy as np
import random
from Games import *

game = TicTacToe()
action_space = game.action_space

def epsilon_greedy(action_space, obs):
    epsilon = 0.1
    if random.random() < 0.1:
        action = np.random.choice(action_space)
    else:
        # FIXME: should choose the best choice through the MCTS
        action = 0
    return action


n_episodes = 100
for i in range(n_episodes):

    # restart the game
    game.restart()

    while not game.terminal:

        # collect observations
        obs = game.board
        action = epsilon_greedy(action_space, obs)
        reward = game.step(action)

        # now will be the turn of the other player
        game.invert_board()