import numpy as np
from Games.Games import Game

class ConnectFour(Game):

    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.uint8)
        self.terminal = False
        action_space = np.arange(0, 7)
        super().__init__(6, 7, 3, action_space, 'ConnectFour')

    def restart(self):
        super().restart()
        self.board = np.zeros((6, 7), dtype=np.uint8)

    def is_valid(self, action):
        col_index = action  # action is in range 0 - 6

        column = self.board[:, col_index]
        i = 0
        while i < len(column) and column[i] == 0:
            i = i + 1
        row_index = i - 1

        if row_index < 0:
            return False
        else:
            return True

    def legal_moves(self):
        legal_moves = []
        for action in self.action_space:
            if self.is_valid(action):
                legal_moves.append(action)
        return legal_moves

    def invert_board(self):
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 1:
                    self.board[row][col] = 2
                elif self.board[row][col] == 2:
                    self.board[row][col] = 1

    def step(self, action):
        """
        Args:
            action(int): a valid action
        Returns:
            reward(int) a integer which is either -1,0, or 1

        self.board    is updated in the process
        self.terminal is updated in the process
        """

        # insert
        col_index = action  # action is in range 0 - 6
        row_index = (self.board[:, col_index] != 0).argmax(
            axis=0) - 1  # subtract one from index of top filled space
        self.board[row_index][col_index] = 1

        # undecided
        terminal = 1

        # to check for 4 in a row horizontal
        for row in range(6):
            a = 0
            for col in range(7):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

        # to check for 4 in a row vertical
        for col in range(7):
            a = 0
            for row in range(6):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

        # diagonal top-left to bottom-right
        for col in range(7):
            a = 0
            row = 0
            while (row < 6) & (col < 7):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

                row = row + 1
                col = col + 1

        for row in range(6):
            a = 0
            col = 0
            while (row < 6) & (col < 7):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

                row = row + 1
                col = col + 1

        # diagonal bottom-left to top-right
        for col in range(7):
            a = 0
            row = 5
            while (row >= 0) & (col < 7):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

                row = row - 1
                col = col + 1

        for row in range(6):
            a = 0
            col = 0
            while (row >= 0) & (col < 7):
                if self.board[row][col] == 1:
                    a = a + 1
                else:
                    a = 0
                if a == 4:
                    self.terminal = True
                    return +1

                row = row - 1
                col = col + 1

        # checks if board is filled completely
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 0:
                    terminal = 0
                    break
        if terminal == 1:
            self.terminal = True

        return 0

    def render(self):
        """
        print to screen the full board nicely
        """

        for i in range(6):
            print('\n|', end="")
            for j in range(7):
                if self.board[i][j] == 1:
                    print(' X |', end="")
                elif self.board[i][j] == 0:
                    print('   |', end="")
                else:
                    print(' O |', end="")
        print('\n')

