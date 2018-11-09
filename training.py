import pickle
import tensorflow
from nn import *
from Games import *

game = GameGlue(TicTacToe())

data = pickle.load(open(game.name, "rb"))

for i in range(len(data)):
    board = data[i][0]
    game.board = board
    layers = game.layers()
    data[i][0] = layers

