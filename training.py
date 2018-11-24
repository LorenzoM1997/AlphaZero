import fnmatch
import os
import pickle
import tensorflow as tf
from nn import *
from Games.Games import Game
from Games.TicTacToe import *
from nn import NN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def find(pattern, path):
    result = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            result.append(file)
    return result


def load_data_for_training(game):

    mypath = 'saved'
    # list of files
    files = find(game.name + '*', mypath)

    X = np.empty((0, game.num_layers, game.num_rows, game.num_cols))  # where input (board state) will be saved
    V = np.empty((0, 1))  # where the value (one of labels) will be saved
    P = []  # where the policy (one of the labels) will be saved

    for file in files:
        print(file)
        try:
            data = pickle.load(open(mypath+'\\'+file, "rb")
                               )  # load the data from file
        except:
            print("Data not found in ", file)
            continue

        X_file = np.empty((0, game.num_layers, game.num_rows, game.num_cols))
        V_file = np.empty((0, 1))
        for episode in data:
            X_episode = np.empty((len(episode), game.num_layers, game.num_rows, game.num_cols))
            V_episode = np.empty((len(episode), 1))
            for i in range(len(episode)):
                game.board = episode[i][0]
                X_episode[i] = game.layers() 
                P.append(episode[i][1])
                V_episode[i] = episode[i][2]

            X_file = np.append(X_file, X_episode, axis = 0)
            V_file = np.append(V_file, V_episode, axis = 0)

        X = np.append(X, X_file, axis = 0)
        V = np.append(V, V_file, axis = 0)
        print("Correctly loaded: ", file)

    print("Episodes in data_set:", len(V))
    return [X, V, P]


def training_nn(game, nnet):
    """
    Args:
        game: a Game object
        nnet: a NN object
    """
    X, V, P = load_data_for_training(game)
    model_path = './model/checkpoint/' + 'model.ckpt'

    nnet.pre_run(model_path)
    vh_pred, ph_pred = nnet.pred([X[0,:,:]])
    print(vh_pred, ph_pred)
    nnet.fit(X, V, P, 64, 100, model_path, model_path)


if __name__ == "__main__":
    # IMPORTANT game definition
    game = TicTacToe()
    input_shape = game.layers().shape
    nnet = NN(input_shape, 5, len(game.action_space), True)
    training_nn(game, nnet)

    
