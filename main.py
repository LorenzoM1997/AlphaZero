from functools import partial
import Games
import UI
from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from Games.Checkers import Checkers
from GameGlue import GameGlue
from UI.GameDisplay import DisplayMain
import multiprocessing
import numpy as np
from nn import NN
import pickle
import progressbar
import random
import settings
import tensorflow as tf
from time import sleep, strftime, gmtime
from training import *
import uct

# change the following line to change game
game_interface = TicTacToe()
game = GameGlue(game_interface)


def random_move():
    global game
    action = np.random.choice(game.legal_moves())
    return [action, np.zeros(len(game.action_space))]


def epsilon_greedy(greedy_move):
    global game
    epsilon = 0.05
    if random.random() < epsilon:
        action, policy = random_move()
    else:
        try:
            action, policy = greedy_move()
        except:
            action = greedy_move()
            policy = np.ones(len(game.action_space))
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


def ai_move(ai, names, inputs, outputs, mode='training'):
    global game
    ai.update(game.state)
    if mode == 'training':
        #  during training we are using some randomization
        action, policy = epsilon_greedy(partial(ai.get_action, names, inputs, outputs))
        if np.all(policy == 1):
            # overwrite the policy if it didn't make a random move
            policy = ai.policy
    else:
        #  in evaluation we are taking the greedy action
        action = ai.get_action(names, inputs, outputs)
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


def elo_rating(results, tasks, scores, elo_opponent=0, main_player=random_move, opponent=random_move):
    reward, episodes = simulation(results, tasks, main_player, opponent,
                                  render=False, save_episodes=True, evaluation=True)
    elo = (reward * 400) / episodes + elo_opponent
    scores.put(elo)


if __name__ == "__main__":

    print("Game: ",game_interface.name)

    ai = uct.UCTValues(game)
    ai_old = uct.UCTValues(game)

    settings.init(game_interface)

    #  modes: 'training', 'manual', 'debug', 'evaluation'
    mode = 'training'

    if mode == 'training':
        render_game = False
        num_episodes = 4
        num_simulations = 1
        episode_to_save = 4
        save_episodes = True
        ai.DEBUG = False
        ai.use_nn = True
        ai_old.DEBUG = False
        ai_old.use_nn = True

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
        save_episodes = True
        ai.DEBUG = True
        ai_old.DEBUG = True

        print("Mode: debug.")
        print("Parallel simulations: ", num_simulations)
        print("Total number of episodes: ", num_simulations * num_episodes)

    elif mode == 'evaluation':
        render_game = False
        num_simulations = 4
        num_episodes = 100
        episode_to_save = 4
        save_episodes = True
        ai.DEBUG = False

        total_episodes = num_simulations * num_episodes  # total number of episodes

        print("Mode: evaluation")
        print("Parallel simulations:", num_simulations)
        print("Total number of episodes:", total_episodes)

        try:
            #  file where the episodes are saved
            filename = 'saved\\' + game.name + \
                strftime("%Y-%m-%d-", gmtime()) + \
                str(np.random.randint(100000))
        except:
            print("[WARNING] Evaluation mode will not be saving files.")

    else:
        print("mode name not recognized.")
        exit()

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()
    scores = manager.Queue()

    command_list = [manager.Queue() for i in range(num_simulations)]
    input_list = [manager.Queue() for i in range(num_simulations)]
    output_list = [manager.Queue() for i in range(num_simulations)]

    if mode == 'manual':
        #  testing manually
        tasks.put(1)
        simulation(results, tasks, render=True, main_player=partial(
            ai_move, ai), opponent=manual_move, save_episodes=save_episodes)

    elif mode == 'debug':

        # create tasks list
        for i in range(num_simulations):
            tasks.put(num_episodes)

        # start simulations processes
        pool = multiprocessing.Pool(processes=num_simulations)
        processes = []
        for i in range(num_simulations):
            new_process = multiprocessing.Process(target=simulation, args=(
                results, tasks, partial(ai_move, ai), partial(ai_move, ai_old), render_game, save_episodes,))
            processes.append(new_process)
            new_process.start()

        while True:
            # Read result
            new_result = results.get()

            if render_game:
                #  call the UI appropiate for the game
                DisplayMain(new_result, game_interface.name)

    elif mode == 'evaluation':

        memory = []
        prev_elo = 0
        # create tasks list
        for i in range(num_simulations):
            tasks.put(num_episodes)

        # start simulations processes
        pool = multiprocessing.Pool(processes=num_simulations)
        processes = []
        for i in range(num_simulations):
            new_process = multiprocessing.Process(target=elo_rating, args=(
                            results, tasks, scores, prev_elo, partial(ai_move, ai), random_move, ))
            processes.append(new_process)
            new_process.start()

        num_finished_simulations = 0
        elo = 0

        bar = progressbar.ProgressBar(maxval=total_episodes,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while True:
            # Read result
            new_result = results.get()

            # increment simulations
            memory.append(new_result)
            num_finished_simulations += 1
            bar.update(num_finished_simulations)

            if num_finished_simulations == total_episodes:

                for s in range(num_simulations):
                    score = scores.get() 
                    elo = elo - score
                elo = elo/ num_simulations
                print('elo:', elo)

                pickle.dump(memory, open(filename, "wb"))

            if num_finished_simulations % episode_to_save == 0 and num_finished_simulations > 0:
                pickle.dump(memory, open(filename, "wb"))
          
        bar.finish()

    elif mode == 'training':
        num_finished_simulations = 0
        training = False
        memory = []
        prev_elo = 360
        elo = 0

        #  start the progressbar
        bar = progressbar.ProgressBar(maxval=total_episodes,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        while True: 

            if not training:

                if num_finished_simulations == 0:

                    # create tasks list
                    for i in range(num_simulations):
                        tasks.put(num_episodes)

                    # restart all simulations
                    pool = multiprocessing.Pool(processes=num_simulations)
                    processes = []
                    for i in range(num_simulations):
                        new_process = multiprocessing.Process(target=elo_rating, args=(
                            results, tasks, scores, prev_elo, partial(ai_move, ai, command_list[i], input_list[i], output_list[i]),
                            partial(ai_move, ai_old, command_list[i], input_list[i], output_list[i]), ))
                        processes.append(new_process)
                        new_process.start()

                num_finished_MCTS = 0
                settings.Trainer.prepare('new')
                while True:
                    try:
                        new_result = results.get(block = False)
                        break
                    except:
                        pass

                    for i in range(len(command_list)):
                        try:
                            name = command_list[i].get(block = False)
                        except:
                            continue
                        if name == 'done':
                            break
                        new_input = input_list[i].get()
                        output = settings.Trainer.pred(new_input)
                        output_list[i].put(output)
                # Save in list
                memory.append(new_result)
                num_finished_simulations += 1
                bar.update(num_finished_simulations)

                #  when all simulations are complete
                if num_finished_simulations == total_episodes:

                    for s in range(num_simulations):
                        score = scores.get() 
                        elo = elo - score
                    elo = elo/ num_simulations
                    print('elo:', elo)

                    # save memory
                    pickle.dump(memory, open(filename, "wb"))

                    # Set process for training the network
                    """
                    if elo > prev_elo:
                        settings.Trainer.train('new')
                    else:
                        settings.Trainer.train('old')
                    """

                    settings.Trainer.train('new')
                    ai.use_nn = True

                    # reset simulations count
                    num_finished_simulations = 0
                    training = True

                 # save memory every n episodes
                if num_finished_simulations % episode_to_save == 0 and num_finished_simulations > 0:
                    pickle.dump(memory, open(filename, "wb"))

            else:

                print("starting training")
                # wait to read result
                new_result = results.get()

                print("completed training")
                training = False

        bar.finish()
