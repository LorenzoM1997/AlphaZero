import numpy as np
from Games.Games import Game

class TicTacToe(Game):
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.uint8)
        self.search_time = 4
        action_space = np.arange(0, 9)
        super().__init__(3, 3, 3, action_space, 'TicTacToe')

    def restart(self):
        super().restart()
        self.board = np.zeros((3, 3), dtype=np.uint8)

    def is_valid(self, action):
        """
        Checks if an action is valid

        Args:
            action(int)
        """
        if self.board[int(np.floor(action / 3))][action % 3] != 0:
            return False
        else:
            return True

    def legal_moves(self):
        """
        return:
            legal_moves(list): a list with all the legal moves from the current position
        """
        legal_moves = []
        for action in self.action_space:
            if self.is_valid(action):
                legal_moves.append(action)
        return legal_moves

    def invert_board(self):
        self.board = (3 - self.board) % 3

    def step(self, action):
        """
        Args:
            action(int): a valid action
        RETURN: reward (-1,0,1)
        self.board    is updated in the process
        self.terminal is updated in the process
        """
        # insert
        row_index = int(np.floor(action / 3))
        col_index = action % 3
        self.board[row_index][col_index] = 1

        # undecided
        terminal = 1

        # to check for 3 in a row horizontal
        if np.all(self.board[row_index] == 1):
            self.terminal = True
            return + 1

        # to check for 3 in a row vertical
        if np.all(self.board[:,col_index] == 1):
            self.terminal = True
            return +1

        # diagonal top-left to bottom-right
        for diag in range(3):
            if self.board[diag][diag] != 1:
                terminal = 0
        if terminal == 1:
            self.terminal = True
            return +1
        else:
            terminal = 1
        # diagonal bottom-left to top-right
        for diag in range(3):
            if self.board[2 - diag][diag] != 1:
                terminal = 0
        if terminal == 1:
            self.terminal = True
            return +1
        else:
            terminal = 1

        # checks if board is filled completely
        if np.any(self.board == 0):
            terminal = 0

        if terminal == 1:
            self.terminal = True
        return 0

    def render(self):
        """
        print to screen the full board nicely
        """
        for i in range(3):
            print('\n|', end="")
            for j in range(3):
                if self.board[i][j] == 1:
                    print(' X |', end="")
                elif self.board[i][j] == 0:
                    print('   |', end="")
                else:
                    print(' O |', end="")
        print('\n')
