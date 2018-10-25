import tensorflow as tf
import numpy as np
import random
from Games import *
from MCTS import *

game = TicTacToe()
action_space = game.action_space


def random_move():
    global game
    action = np.random.choice(game.action_space)
    while not game.is_valid(action):
        action = np.random.choice(game.action_space)
    return action


def epsilon_greedy(greedy_move):
    epsilon = 0.1
    if random.random() < 0.1:
        action = random_move()
    else:
        action = greedy_move()
    return action


def simulation(n_episodes=100, opponent=random_move):
    for i in range(n_episodes):

        # restart the game
        game.restart()
        player = n_episodes % 2

        while not game.terminal:

            game.render()

            # collect observations
            if player:
                # FIXME: should choose the best choice through the MCTS
                action = epsilon_greedy(random_move)
            else:
                action = opponent()
            reward = game.step(action)

            # now will be the turn of the other player
            game.invert_board()


# uncomment this when MCTS ready
# mct = MCT(game)

simulation()
