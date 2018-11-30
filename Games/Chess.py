import math
import copy
from functools import reduce
import numpy as np
from Games.Games import Game


class Chess(Game):
    EMPTY_SPOT = 0

    P1_King = 1
    P1_Queen = 2
    P1_Bishop = 3
    P1_Knight = 4
    P1_Rook = 5
    P1_Pawn = 6
    P2_King = 7
    P2_Queen = 8
    P2_Bishop = 9
    P2_Knight = 10
    P2_Rook = 11
    P2_Pawn = 12

    HEIGHT = 8
    WIDTH = 8
    MAX_MOVES = 150

    def __init__(self, old_spots=None):
        """
        Unless specified otherwise,
        the board will be created with a start board configuration.
        the_player_turn=True indicates turn of player P1
        """
        # initialize player and moves
        self.player_turn = True  # true for player one
        self.moves_taken = 0

        # initialize board
        if old_spots is None:
            spots = np.concatenate((np.array([P1_Rook],[P1_Knight],[P1_Bishop],[P1_Queen],[P1_King],[P1_Bishop],[P1_Knight],[P1_Rook]), 
                    np.full((8,1),P1_Pawn),np.full((8,4),EMPTY_SPOT),np.full((8,1),P2_Pawn),
                    np.array([P2_Rook],[P2_Knight],[P2_Bishop],[P2_Queen],[P2_King],[P2_Bishop],[P2_Knight],[P2_Rook])),axis=None)

            self.board = np.rot90(self.board, 1)
        else:
            self.board = np.array(old_spots)

        # initialize action space
        action_space = np.zeros((64, 64, 2), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                action_space[i, j] = np.array([i, j])

        super().__init__(8, 8, 2, action_space, 'Chess')


    # def tile_to_row_col(self, tile):
    #     """
    #     Converts tile index to row col index.
    #     """
    #     row = np.floor(tile / self.WIDTH)
    #     col = tile % self.WIDTH
    #     return np.array([row, col])


    # def row_col_to_tile(self, row_col):
    #     """
    #     Converts row col index to tile index.
    #     """
    #     row = row_col[0]
    #     col = row_col[1]
    #     tile = self.WIDTH * row + col
    #     return tile


    # def action_to_move(self, action):
    #     """
    #     Converts an action from the action space to a 2 by 2 array representing
    #     a move.
    #     """
    #     return np.array([self.tile_to_row_col(action[0]),
    #                      self.tile_to_row_col(action[1])], dtype=np.uint8)

    # def move_to_action(self, move):
    #     """
    #     Converts a 2 by 2 array move to an action.
    #     """
    #     return np.array([self.row_col_to_tile(move[0]),
    #                       self.row_col_to_tile(move[1])], dtype=np.uint8)

    def restart(self):
        """
        Resets the game to the starting position.
        """
        self.board = Chess().board


    def invert_board(self):
        """
        Inverts the positions of the two players.
        """
        new_board = np.zeros(self.board.shape)
        for i in range(1,13):
            if i < 7:
                new_board[self.board==i] = i+6
            else:
                new_board[self.board==i] = i-6

        self.board = new_board
        self.board = np.rot90(self.board, 2)


    # def not_spot(self, loc):
    #     """
    #     Determines if a location is within the bounds of the board.
    #     """
    #     if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or \
    #         loc[1] > self.WIDTH - 1:
    #         return True
    #     return False


    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        """
        return self.board[loc[0]][loc[1]]


    def check_win_conditions(self):
        """
        Returns true when a player wins (one of the players is missing their king)
        or draw.
        """
        # Check for a win
        King_1 = 0
        King_2 = 0
        for i in range(32):
            for j in range(32):
                if self.board[i][j] == P1_King
                    King_1 = 1
                    break
            else:
                continue
            break

        for i in range(32):
            for j in range(32):
                if self.board[i][j] == P2_King
                    King_2 = 1
                    break
            else:
                continue
            break

        #Check if Player 1 wins
        if King_1 == 1 && King_2 == 0
            return 1

        #Check if Player 2 wins
        if King_1 == 0 && King_2 == 1
            return -1

        #Check if there is a draw - there are no legal moves available
        if len(legal_moves) == 0
            return 0  

    return 0

    #The king can move one square in any direction.
    def king_moves(self,row,col):
        moves = []

        for i in [row-1,row,row+1]:
            for k in [col-1,col,col+1]:
                if i!=row && j!=col
                moves = moves + (i,j)


    def queen_moves(self,i,j):

    def bishop_moves(self,i,j):

    def knight_moves(self,i,j):

    def rook_moves(self,i,j):

    def pawn_moves(self,i,j):





    def legal_moves(self):
        """
        Returns all of the legal possible moves for Player 1.
        """
        actions = []

        for i in range(32):
            for j in range(32):
                if self.board[i][j] == P1_King
                    actions = actions + king_moves(i,j)
                if self.board[i][j] == P1_Queen
                    actions = actions + queen_moves(i,j)
                if self.board[i][j] == P1_Bishop
                    actions = actions + bishop_moves(i.j)
                if self.board[i][j] == P1_Knight
                    actions = actions + knight_moves(i,j)
                if self.board[i][j] == P1_Rook
                    actions = actions + rook_moves(i,j)
                if self.board[i,j] == P1_Pawn
                    actions = actions + pawn_moves(i,j)

        for i in range(len(moves)):
            actions[i] = self.move_to_action(moves[i])
        return actions.tolist()


    def step(self, action):
        """
        Makes a given move on the board, and (as long as is wanted) switches the indicator for which players turn it is.
        """
        if self.legal_moves():
            return -1
        move = self.action_to_move(action)
        if abs(move[0][0] - move[1][0]) == 2:
            for j in range(len(move) - 1):
                if move[j][0] % 2 == 1:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j][1]
                    else:
                        middle_y = move[j + 1][1]
                else:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j + 1][1]
                    else:
                        middle_y = move[j][1]

                self.board[int((move[j][0] + move[j + 1][0]) / 2)][middle_y] = self.EMPTY_SPOT

        self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.board[move[0][0]][move[0][1]]
        if move[len(move) - 1][0] == self.HEIGHT - 1 and self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P1:
            self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P1_K
        elif move[len(move) - 1][0] == 0 and self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P2:
            self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P2_K
        else:
            self.board[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.board[move[0][0]][move[0][1]]
        self.board[move[0][0]][move[0][1]] = self.EMPTY_SPOT

        self.moves_taken += 1
        if self.legal_moves():
            return -1

        return self.check_win_conditions()


    def get_symbol(self, location):
        """
        Gets the symbol for what should be at a board location.
        """
        if self.board[location[0]][location[1]] == self.EMPTY_SPOT:
            return " "
        elif self.board[location[0]][location[1]] == self.P1_King:
            return 'k'
        elif self.board[location[0]][location[1]] == self.P1_Queen:
            return 'q'
        elif self.board[location[0]][location[1]] == self.P1_Bishop:
            return 'b'
        elif self.board[location[0]][location[1]] == self.P1_Knight:
            return '*'
        elif self.board[location[0]][location[1]] == self.P1_Rook:
            return 'r'
        elif self.board[location[0]][location[1]] == self.P1_Pawn:
            return 'p'
        elif self.board[location[0]][location[1]] == self.P2_King:
            return 'K'
        elif self.board[location[0]][location[1]] == self.P2_Queen:
            return 'Q'
        elif self.board[location[0]][location[1]] == self.P2_Bishop:
            return 'B'
        elif self.board[location[0]][location[1]] == self.P2_Knight:
            return '+'
        elif self.board[location[0]][location[1]] == self.P2_Rook:
            return 'R'
        else:
            return 'P'



    def render(self):
        """
        Prints a string representation of the current game board.
        """

        index_columns = "   "
        for j in range(self.WIDTH):
            index_columns += " " + str(j) + "   " + str(j) + "  "
        print(index_columns)

        norm_line = "  |---|---|---|---|---|---|---|---|"
        print(norm_line)

        for j in range(self.HEIGHT):
            temp_line = str(j) + " "
            if j % 2 == 1:
                temp_line += "|///|"
            else:
                temp_line += "|"
            for i in range(self.WIDTH):
                temp_line = temp_line + " " + self.get_symbol([j, i]) + " |"
                if i != 3 or j % 2 != 1:
                    temp_line = temp_line + "///|"
            print(temp_line)
            print(norm_line)
