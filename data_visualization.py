import numpy as np
from training import *
from Games.Games import Game
from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from Games.Checkers import Checkers
import matplotlib.pyplot as plt

if __name__ == "__main__":
    game = TicTacToe()

    X,V,P = load_data_for_training(game)
    
    for i in range(10):
        print(X[i])
        print(V[i])
        print(P[i])

    plt.hist(V, bins = 3)
    plt.show()

    for i in range(P.shape[1]):
        plt.hist(P[:,i], bins = 50)
        plt.show()
