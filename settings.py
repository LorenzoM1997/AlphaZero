from training import NetTrainer
from training import training_nn
import numpy as np
from uct import UCTValues
from GameGlue import GameGlue
import os
from time import strftime, gmtime
from os import path


def init(game_interface):
    global Trainer
    global ai
    global ai_old
    global name_game

    name_game = game_interface.name
    Trainer = NetTrainer(game_interface)
    ai = UCTValues(GameGlue(game_interface))
    ai_old = UCTValues(GameGlue(game_interface))


def set_mode(mode, num_simulations, total_episodes):

    global save_episodes
    global num_episodes
    global render_game
    global episode_to_save
    global filename

    num_episodes = int(np.ceil(total_episodes / num_simulations))
    episode_to_save = 10

    # create the directory if it doesn't exist
    if not path.isdir('saved'):
        os.makedirs('saved')

    try:
        #  file where the episodes are saved
        generated_name = name_game + strftime("%Y-%m-%d-", gmtime()) + \
            str(np.random.randint(100000))
        filename = path.join('saved', generated_name)
    except:
        print("Directory not found")

    if mode == 'training':
        render_game = False
        save_episodes = True
        ai.DEBUG = False
        ai.use_nn = False
        ai_old.DEBUG = False
        ai_old.use_nn = False

        print("Mode: training.")
        print("Parallel simulations: ", num_simulations)
        print("Total number of episodes: ", num_simulations * num_episodes)

    elif mode == 'manual':
        render_game = True
        save_episodes = False
        ai.DEBUG = False

        print("Mode: manual.")

    elif mode == 'debug':
        render_game = True
        save_episodes = True
        ai.DEBUG = True
        ai_old.DEBUG = True

        print("Mode: debug.")
        print("Parallel simulations: ", num_simulations)
        print("Total number of episodes: ", num_simulations * num_episodes)

    elif mode == 'evaluation':
        render_game = False
        save_episodes = True
        ai.DEBUG = False

        print("Mode: evaluation")
        print("Parallel simulations:", num_simulations)
        print("Total number of episodes:", total_episodes)

    else:
        print("mode name not recognized.")
        exit()
