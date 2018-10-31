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


def manual_move():
    global game
    try:
        action = int(input())
    except:
        print("enter a number")
        action = -1
    while not game.is_valid(action):
        try:
            action = int(input())
        except:
            print("enter a number")
            action = -1
    return action

def simulation(n_episodes=100, opponent=random_move, render = True):
    for i in range(n_episodes):

        # restart the game
        game.restart()
        player = i % 2

        while not game.terminal:

            if render:
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
            player = (player + 1) % 2


# uncomment this when MCTS ready
# mct = MCT(game)

# test
simulation()


# manual testing
simulation(n_episodes= 1, opponent = manual_move)
