import pickle
import tensorflow
from nn import *
from Games import *

def load_data_for_training(game):

    data = pickle.load(open(game.name, "rb")) #load the data from file
    X = [] # where input (board state) will be saved
    V = [] # where the value (one of labels) will be saved
    P = [] # where the policy (one of the labels) will be saved

    for episode in data:
        for step in episode:
            game.board = step[0]
            layers = game.layers()
            X.append(layers)
            V.append(step[1])
            P.append(step[2])

    print("Correctly loaded data set")
    print("Episodes in data_set:", len(data))

if __name__== "__main__":
    # IMPORTANT game definition
    game = TicTacToe()
    load_data_for_training(game)