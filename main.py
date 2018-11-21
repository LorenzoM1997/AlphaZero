from functools import partial
import Games
import UI
from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from GameGlue import GameGlue
from UI.GameDisplay import DisplayMain
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


def random_move():
    global game
    action = np.random.choice(game.legal_moves())
    return [action, np.zeros(len(game.action_space))]


def epsilon_greedy(greedy_move):
    epsilon = 0.05
    if random.random() < epsilon:
        action, policy = random_move()
    else:
        try:
            action, policy = greedy_move()
        except:
            action = greedy_move()
            policy = -1
    return [action, policy]


def manual_move():
    global game
    try:
        action = int(input())
    except:
        print("enter a number")
        action = -1
    while action not in game.legal_moves():
        try:
            action = int(input())
        except:
            print("enter a number")
            action = -1
    return [action, np.zeros(len(game.action_space))]


def ai_move(ai, mode='training'):
    global game
    ai.update(game.state)
    if mode == 'training':
        #  during training we are using some randomization
        action, policy = epsilon_greedy(ai.get_action)
        if policy == -1:
            # overwrite the policy if it didn't make a random move
            policy = ai.policy
    else:
        #  in evaluation we are taking the greedy action
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

        #  restart the game
        game.restart()
        episode = []
        player = i % 2

        while not game.terminal:

            if render:
                game_interface.render()

            #  collect observations
            if player:
                action, policy = main_player()
            else:
                action, policy = opponent()

            if save_episodes:
                tuple = [game.board, policy, 0]
                episode.append(tuple)
            reward = game.step(action)

            #  now will be the turn of the other player
            game.invert_board()
            player = (player + 1) % 2

        if evaluation:
            if player:
                total_reward += reward
            else:
                total_reward -= reward

        if save_episodes:
            #  backpropagate the reward
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

    #  modes: 'training', 'manual', 'debug'
    mode = 'debug'

    if mode == 'training':
        render_game = False
        num_episodes = 25
        num_simulations = 4
        episode_to_save = 5
        save_episodes = True
        ai.DEBUG = False
        ai_old.DEBUG = False

        print("Mode: training.")
        print("Parallel simulations: ", num_simulations)
        print("Total number of episodes: ", num_simulations * num_episodes)

        total_episodes = num_simulations * num_episodes  # total number of episodes

        try:
            #  file where the episodes are saved
            filename = 'saved\\' + game.name + \
                strftime("%Y-%m-%d-", gmtime()) + \
                str(np.random.randint(100000))
        except:
            print("Directory not found")
            exit()

    elif mode == 'manual':
        render_game = True
        num_simulations = 0
        num_episodes = 10
        save_episodes = False
        ai.DEBUG = False

        print("Mode: manual.")

    elif mode == 'debug':
        render_game = True
        num_simulations = 1
        num_episodes = 25
        save_episodes = False
        ai.DEBUG = True

        print("Mode: debug.")
        print("Parallel simulations: ", num_simulations)
        print("Total number of episodes: ", num_simulations * num_episodes)

    else:
        print("mode name not recognized.")
        exit()

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

    # UNCOMMENT THIS for testing the ELO rating
    # tasks.put(100)
    #print("ELO rating against random: ", elo_rating(results, tasks, partial(ai_move, ai)))

    # Set process for training the network
    new_process = multiprocessing.Process(
        target=load_data_for_training, args=(game_interface,))
    processes.append(new_process)
    new_process.start()
    if mode == 'manual':
        #  testing manually
        tasks.put(1)
        simulation(results, tasks, render=True, main_player=partial(
            ai_move, ai), opponent=manual_move, save_episodes=save_episodes)

    elif mode == 'debug':

        while True:
            # Read result
            new_result = results.get()

    elif mode == 'training':
        num_finished_simulations = 0
        memory = []

        #  start the progressbar
        bar = progressbar.ProgressBar(maxval=total_episodes,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while True:

            # save memory every n episodes
            if num_finished_simulations % episode_to_save == 0 and num_finished_simulations > 0:
                pickle.dump(memory, open(filename, "wb"))

            # Read result
            new_result = results.get()

            # Save in list
            memory.append(new_result)
            num_finished_simulations += 1
            bar.update(num_finished_simulations)

            if render_game:
                #  call the UI appropiate for the game
                DisplayMain(new_result, game_interface.name)

            if num_finished_simulations == total_episodes:
                #  when all the simulations are completed save and exit
                pickle.dump(memory, open(filename, "wb"))
                break

        bar.finish()
