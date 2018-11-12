from functools import partial
import Games
from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from GameGlue import GameGlue
import multiprocessing
import numpy as np
import pickle
import progressbar
import random
import tensorflow as tf
from time import sleep, strftime, gmtime
from training import load_data_for_training
import uct

# change the following line to change game
game_interface = ConnectFour()
game = GameGlue(game_interface)
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


def simulation(results, n_episodes=100, opponent=random_move,
               render=True, save_episodes=False, evaluation=False):

    global game

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
            results.put(episode)

    if evaluation:
        return total_reward


def elo_rating(results, elo_opponent=0, episodes=100, opponent=random_move):
    reward = simulation(results, episodes, random_move,
                        render=False, evaluation=True)
    elo = (reward * 400) / episodes + elo_opponent
    return elo


if __name__ == "__main__":

    render_game = False
    save_episodes = True
    num_episodes = 70

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    results = manager.Queue()

    # Create process pool with two processes
    num_simulations = 3
    num_processes = 1 + num_simulations
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    for i in range(num_simulations):
        # Initiate the worker processes for simulation
        new_process = multiprocessing.Process(target=simulation, args=(
            results, num_episodes, partial(ai_move, ai), render_game, save_episodes,))
        processes.append(new_process)
        new_process.start()

    # UNCOMMENT THIS for testing manually
    # simulation(results, 10, render=True, opponent=manual_move, save_episodes=True)

    # UNCOMMENT THIS for testing the ELO rating
    # print("ELO rating against random: ", elo_rating(results, episodes = 10))

    # Set process for training the network
    new_process = multiprocessing.Process(
        target=load_data_for_training, args=(game_interface,))
    processes.append(new_process)
    new_process.start()

    if save_episodes:
        num_finished_simulations = 0
        memory = []

        # calculate total number of episodes
        total_episodes = num_simulations * num_episodes

         # progressbar
        bar = progressbar.ProgressBar(maxval= total_episodes, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while True:
            # Read result
            new_result = results.get()
            # Save in list
            memory.append(new_result)
            num_finished_simulations += 1
            bar.update(num_finished_simulations)

            if num_finished_simulations == total_episodes:
                pickle.dump(results, open(
                    game.name + strftime("%Y-%m-%d %H:%M", gmtime()), "wb"))
                break

        bar.finish()
