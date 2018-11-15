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
game_interface = TicTacToe()
game = GameGlue(game_interface)

def random_move():
    global game
    action = np.random.choice(game.action_space)
    while not game.is_valid(action):
        action = np.random.choice(game.action_space)
    return [action, np.zeros(len(game.action_space))]


def epsilon_greedy(greedy_move):
    epsilon = 0.1
    if random.random() < 0.1:
        action, policy = random_move()
    else:
        action, policy = greedy_move()
    return [action, policy]


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
    return [action, np.zeros(len(game.action_space))]


def ai_move(ai):
    global game
    ai.update(game.state)
    action = ai.get_action()
    policy = ai.policy
    return [action, policy]


def simulation(results, tasks, main_player=random_move, opponent=random_move,
               render=True, save_episodes=False, evaluation=False):

    global game

    n_episodes = tasks.get()

    if evaluation:
        total_reward = 0

    for i in range(n_episodes):

        # restart the game
        game.restart()
        episode = []
        player = i % 2

        while not game.terminal:

            if render:
                game_interface.render()

            # collect observations
            if player:
                action, policy = main_player()
            else:
                action, policy = opponent()

            if save_episodes:
                tuple = [game.board, policy, 0]
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
        return total_reward, n_episodes


def elo_rating(results, tasks, elo_opponent=0, main_player=random_move, opponent=random_move):
    reward, episodes = simulation(results, tasks, main_player, opponent,
                        render=False, evaluation=True)
    elo = (reward * 400) / episodes + elo_opponent
    return elo


if __name__ == "__main__":

    ai = uct.UCTValues(game)
    ai_old = uct.UCTValues(game)

    # variables
    render_game = False
    save_episodes = True
    num_episodes = 50
    episode_to_save = 10
    num_simulations = 4
    filename = 'saved\\' + game.name + strftime("%Y-%m-%d", gmtime()) + str(np.random.randint(10000))

    # calculate total number of episodes
    total_episodes = num_simulations * num_episodes

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()

    # Create process pool with two processes
    num_processes = 1 + num_simulations
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    for i in range(num_simulations):
        tasks.put(num_episodes)

    for i in range(num_simulations):
        # Initiate the worker processes for simulation
        new_process = multiprocessing.Process(target=simulation, args=(
            results, tasks, partial(ai_move, ai), partial(ai_move, ai_old), render_game, save_episodes,))
        processes.append(new_process)
        new_process.start()

    # UNCOMMENT THIS for testing manually
    #tasks.put(1)
    #simulation(results, tasks, render=True, main_player= partial(ai_move, ai), opponent=manual_move, save_episodes=True)

    # UNCOMMENT THIS for testing the ELO rating
    #tasks.put(100)
    #print("ELO rating against random: ", elo_rating(results, tasks, partial(ai_move, ai)))

    # Set process for training the network
    new_process = multiprocessing.Process(
        target=load_data_for_training, args=(game_interface,))
    processes.append(new_process)
    new_process.start()

    if save_episodes:
        num_finished_simulations = 0
        memory = []

         # progressbar
        bar = progressbar.ProgressBar(maxval= total_episodes, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while True:

            print(num_finished_simulations)
            # save memory every 10 episodes
            if num_finished_simulations % episode_to_save == 0 and num_finished_simulations > 0:
                pickle.dump(memory, open(filename, "wb"))

            # Read result
            new_result = results.get()
            # Save in list
            memory.append(new_result)
            num_finished_simulations += 1
            bar.update(num_finished_simulations)

            if num_finished_simulations == total_episodes:
                pickle.dump(memory, open(filename, "wb"))
                break

        bar.finish()
