import pickle
import tensorflow as tf
from nn import *
from Games.Games import *
from Games.TicTacToe import *
from nn import NN

def load_data_for_training(game):

    try:
        data = pickle.load(open(game.name, "rb"))  # load the data from file
    except:
        print("Data not found.")
        return None

    X = []  # where input (board state) will be saved
    V = []  # where the value (one of labels) will be saved
    P = []  # where the policy (one of the labels) will be saved

    for episode in data:
        for step in episode:
            game.board = step[0]
            layers = game.layers()
            X.append(layers)
            V.append(step[1])
            P.append(step[2])

    print("Correctly loaded data set")
    print("Episodes in data_set:", len(data))
    return [X, V, P]

def training_nn(game, nnet):
    """
    Args:
        game: a Game object
        nnet: a NN object
    """
    X,V,P = load_data_for_training(game)
    model_path = './model/checkpoint/' + 'model.ckpt'
    nnet.fit(self, X, V, P, 32, saver_path = model_path)

if __name__ == "__main__":
    # IMPORTANT game definition
    game = TicTacToe()
    input_shape = game.layers().shape
    nnet = NN(input_shape, 5, game.action_space, True )
    training_nn(game, nnet)
