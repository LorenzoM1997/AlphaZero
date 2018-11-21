import sys
import numpy as np


class Game:

    def __init__(self, num_rows, num_cols, num_layers,
                 action_space, name='undefined'):
        """
        Args:
            num_rows(int): number of rows
            num_cols(int): number of columns
            num_layers(int): number of layers in the layer representation of the board
            action_space(list of integers): all the possible moves
            name(string - optional)
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_layers = num_layers
        self.action_space = action_space
        self.obs_space = num_rows * num_cols * num_layers
        self.terminal = False
        self.name = name

    def layers(self):
        """
        Converts the board into the layers representation
        Useful for the neural network

        returns:
            layers(np.ndarray): a matrix with one-hot encoded positions

        """
        layers = np.zeros((self.num_layers, self.num_rows,
                           self.num_cols), dtype=np.uint8)
        for k in range(self.num_layers):
            layers[k] = np.isin(self.board, k + 1)
        return layers

    def restart(self):
        self.terminal = False
