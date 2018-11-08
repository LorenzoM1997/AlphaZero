import tensorflow as tf
import numpy as np
import random
import pickle
from Games import *
from GameGlue import GameGlue
from functools import partial
import uct

game = GameGlue(TicTacToe())
ai = uct.UCTValues(game)

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
    except BaseException:
        print("enter a number")
        action = -1
    while not game.is_valid(action):
        try:
            action = int(input())
        except BaseException:
            print("enter a number")
            action = -1
    return action

def ai_move(ai):
    global game
    ai.update(game.state)
    return ai.get_action()

def simulation(n_episodes=100, opponent=random_move,
               render=True, save_episodes=False, evaluation=False):

    if save_episodes:
        memory = []

    if evaluation:
        total_reward = 0

    for i in range(n_episodes):

        # restart the game
        game.restart()
        episode = []
        player = i % 2

        while not game.terminal:

            if render:
                game.render()

            # collect observations
            if player:
                # FIXME: should choose the best choice through the MCTS
                ai.update(game.state)
                action = ai.get_action()

            else:
                action = opponent()

            if save_episodes:
                tuple = [game.board, action, 0]
                episode.append(tuple)
            reward = game.step(action)

            # now will be the turn of the other player
            game.invert_board()
            player = (player + 1) % 2

        if evaluation:
            if player:
                total_reward += reward
            else:
                total_reward -= reward

        if save_episodes:
            for i in range(len(episode)):
                episode[len(episode) - i - 1][2] = reward
                reward = reward * (-1)
            memory.append(episode)

    if save_episodes:
        pickle.dump(memory, open(game.name, "wb"))
        if evaluation:
            return [memory, total_reward]
    elif evaluation:
        return total_reward


def elo_rating(elo_opponent=0, episodes=100, opponent=random_move):
    reward = simulation(episodes, random_move, render=False, evaluation=True)
    elo = (reward * 400) / episodes + elo_opponent
    return elo


# UNCOMMENT THIS for testing manually
# simulation(10, render=True, opponent=manual_move, save_episodes=True)

# UNCOMMENT THIS for testing the ELO rating
# print("ELO rating against random: ", elo_rating(episodes = 10))

print("Saving dataset")
memory = simulation(200, partial(ai_move, ai), render= False, save_episodes = True)