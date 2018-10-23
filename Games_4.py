import numpy as np


class Game:

    def __init__(self, num_rows, num_cols, num_layers, action_space):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_layers = num_layers
        self.action_space = action_space
        self.obs_space = num_rows * num_cols * num_layers


class ConnectFour(Game):

    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.uint8)
        self.terminal = False
        action_space = np.arange(0, 42)
        super().__init__(6, 7, 3, action_space)

    def restart(self):
        self.board = np.zeros((6, 7), dtype=np.uint8)
        self.terminal = False

    def is_valid(self, action):
        if self.board[int(np.floor(action / 6))][action % 7] != 0:
            return False
        else:
            return True

    def invert_board(self):
        for row in range(6):
            for col in range(7):
                if(self.board[row][col] == 1):
                    self.board[row][col] = 2
                elif(self.board[row][col] == 2):
                    self.board[row][col] = 1

    def step(self, action):
        """
        PARAMS: a valid action (int 0 to 41)
        RETURN: reward (-1,0,1)
        self.board    is updated in the process
        self.terminal is updated in the process
        """

        # insert
        row_index = int(np.floor(action / 6))
        col_index = action % 7
        self.board[row_index][col_index] = 1

        # undecided
        terminal = 1

        # to check for 4 in a row horizontal
        for row in range(6):
        	a = 0
            for col in range(7):
                if(self.board[row][col] == 1):
                	a = a + 1
                else:
                	a = 0
            	if(a == 4):
                	self.terminal = True
                	return +1

        # to check for 4 in a row vertical
        for col in range(7):
        	a = 0
            for row in range(6):
                if(self.board[row][col] == 1):
                    a = a + 1
                else:
                	a = 0
            	if(a == 4):
                	self.terminal = True
                	return +1

        # diagonal top-left to bottom-right
        for col in range(7):
        	a = 0
        	row = 0
        	while (row < 6) & (col < 7):
        		if(self.board[row][col] == 1):
        			a = a + 1
        		else:
        			a = 0
        		if(a == 4):
                	self.terminal = True
                	return +1
                row = row + 1
                col = col + 1

        for row in range(6):
        	a = 0
        	col = 0
        	while (row < 6) & (col < 7):
        		if(self.board[row][col] == 1):
        			a = a + 1
        		else:
        			a = 0
        		if(a == 4):
                	self.terminal = True
                	return +1
                row = row + 1
                col = col + 1


        # diagonal bottom-left to top-right
		for col in range(7):
        	a = 0
        	row = 5
        	while (row >= 0) & (col < 7):
        		if(self.board[row][col] == 1):
        			a = a + 1
        		else:
        			a = 0
        		if(a == 4):
                	self.terminal = True
                	return +1
                row = row - 1
                col = col + 1

        for row in range(6):
        	a = 0
        	col = 0
        	while (row >= 0) & (col < 7):
        		if(self.board[row][col] == 1):
        			a = a + 1
        		else:
        			a = 0
        		if(a == 4):
                	self.terminal = True
                	return +1
                row = row - 1
                col = col + 1

        # checks if board is filled completely
        for row in range(6):
            for col in range(7):
                if(self.board[row][col] == 0):
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