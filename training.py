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

        nnet.fit(X, V, P, 64, 100, model_path, model_path)

class NetTrainer():

    def __init__(self, game, residual_layers = 5):
        """
        Args:
            game: A Game object
            residual_layers(int): number of residual layers. Default is 5
        """
        self.game = game
        input_shape = game.layers().shape
        policy_shape = len(game.action_space)

        self.nnet_1 = NN(input_shape, residual_layers, policy_shape, True)
        self.path_1 = './model/checkpoint/' + 'old.ckpt'
        self.nnet_2 = NN(input_shape, residual_layers, policy_shape, True)
        self.path_2 = './model/checkpoint/' + 'new.ckpt'

    def train(self, name):
        """
        Args:
            name(string): 'new' or 'old'
        """
        if name == 'old':
            training_nn(self.game, self.nnet_1)
        elif name == 'new':
            training_nn(self.game, self.nnet_2)
        else:
            print("invalid name.")

        # already prepare for evaluation
        self.nnet_2.pre_run(model_path)
        self.nnet_1.pre_run(model_path)

    def pred(self, name, new_input):
        """
        Args:
            name(string): 'new' or 'old'
            new_input: a list [X, V, P]
        """
        if name == 'old':
            self.nnet_1.pred(new_input)
        elif name == 'new':
            self.nnet_2.pred(new_input)
        else:
            print("invalid name.")

if __name__ == "__main__":
    # IMPORTANT game definition
    game = TicTacToe()
    input_shape = game.layers().shape
    nnet = NN(input_shape, 5, len(game.action_space), True)
    training_nn(game, nnet)

    
