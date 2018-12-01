import math
import copy
from functools import reduce
import numpy as np
from Games.Games import Game


class Checkers(Game):
    EMPTY_SPOT = 0
    P1 = 1
    P2 = 2
    P1_K = 3
    P2_K = 4
    BACKWARDS_PLAYER = P2
    HEIGHT = 8
    WIDTH = 4
    MAX_MOVES = 150

    P1_SYMBOL = 'o'
    P1_K_SYMBOL = 'O'
    P2_SYMBOL = 'x'
    P2_K_SYMBOL = 'X'


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
            spots = [[j, j, j, j] for j in [self.P1, self.P1, self.P1, self.EMPTY_SPOT,
                                            self.EMPTY_SPOT, self.P2, self.P2, self.P2]]
        else:
            spots = old_spots
        self.board = np.array(spots)

        # initialize action space
        action_space = np.zeros((32, 32, 2), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                action_space[i, j] = np.array([i, j])

        super().__init__(8, 4, 2, action_space, 'Checkers')


    def tile_to_row_col(self, tile):
        """
        Converts tile index to row col index.
        """
        row = np.floor(tile / self.WIDTH)
        col = tile % self.WIDTH
        return np.array([row, col])


    def row_col_to_tile(self, row_col):
        """
        Converts tile index to row col index.
        """
        row = row_col[0]
        col = row_col[1]
        tile = self.WIDTH * row + col
        return tile


    def action_to_move(self, action):
        """
        Converts an action from the action space to a 2 by 2 array representing
        a move.
        """
        return np.array([self.tile_to_row_col(action[0]),
                         self.tile_to_row_col(action[1])], dtype=np.uint8)

    def move_to_action(self, move):
        """
        Converts a 2 by 2 array move to an action.
        """
        return np.array([self.row_col_to_tile(move[0]),
                          self.row_col_to_tile(move[1])], dtype=np.uint8)


    def restart(self):
        """
        Resets the current configuration of the game board to the original
        starting position.
        """
        self.board = Checkers().board


    def invert_board(self):
        """
        Toggles the player.
        """
        new_board = np.zeros(self.board.shape)
        new_board[self.board==1] = 2
        new_board[self.board==2] = 1
        new_board[self.board==3] = 4
        new_board[self.board==4] = 3
        self.board = new_board
        self.board = np.rot90(self.board, 2)


    def not_spot(self, loc):
        """
        Finds out of the spot at the given location is an actual spot on the game board.
        """
        if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or \
            loc[1] > self.WIDTH - 1:
            return True
        return False


    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        """
        return self.board[loc[0]][loc[1]]


    def forward_n_locations(self, start_loc, n, backwards=False):
        """
        Gets the locations possible for moving a piece from a given location diagonally
        forward (or backwards if wanted) a given number of times(without directional change midway).
        """
        if n % 2 == 0:
            temp1 = 0
            temp2 = 0
        elif start_loc[0] % 2 == 0:
            temp1 = 0
            temp2 = 1
        else:
            temp1 = 1
            temp2 = 0

        answer = [[start_loc[0], start_loc[1] + math.floor(n / 2) + temp1],
                    [start_loc[0], start_loc[1] - math.floor(n / 2) - temp2]]

        if backwards:
            answer[0][0] = answer[0][0] - n
            answer[1][0] = answer[1][0] - n
        else:
            answer[0][0] = answer[0][0] + n
            answer[1][0] = answer[1][0] + n

        if self.not_spot(answer[0]):
            answer[0] = []
        if self.not_spot(answer[1]):
            answer[1] = []

        return answer


    def get_simple_moves(self, start_loc):
        """
        Gets the possible moves a piece can make given that it does not capture any
        opponents pieces.
        """
        if self.board[start_loc[0]][start_loc[1]] == 2:
            next_locations = self.forward_n_locations(start_loc, 1)
            next_locations.extend(self.forward_n_locations(start_loc, 1, True))
        else:
            next_locations = self.forward_n_locations(start_loc, 1)


        possible_next_locations = []

        for location in next_locations:
            if len(location) != 0:
                if self.board[location[0]][location[1]] == self.EMPTY_SPOT:
                    possible_next_locations.append(location)

        return [[start_loc, end_spot] for end_spot in possible_next_locations]


    def get_capture_moves(self, start_loc, move_beginnings=None):
        """
        Recursively get all of the possible moves for a piece which involve capturing an
        opponent's piece.
        """
        if move_beginnings is None:
            move_beginnings = [start_loc]

        answer = []
        if self.board[start_loc[0]][start_loc[1]] > 2:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
            next1.extend(self.forward_n_locations(start_loc, 1, True))
            next2.extend(self.forward_n_locations(start_loc, 2, True))
        elif self.board[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next1 = self.forward_n_locations(start_loc, 1, True)
            next2 = self.forward_n_locations(start_loc, 2, True)
        else:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)


        for j in range(len(next1)):
            # if both spots exist
            if (not self.not_spot(next2[j])) and (not self.not_spot(next1[j])) :
                # if next spot is opponent
                if self.get_spot_info(next1[j]) != self.EMPTY_SPOT and \
                    self.get_spot_info(next1[j]) % 2 != self.get_spot_info(start_loc) % 2:
                    # if next next spot is empty
                    if self.get_spot_info(next2[j]) == self.EMPTY_SPOT:
                        temp_move1 = copy.deepcopy(move_beginnings)
                        temp_move1.append(next2[j])

                        answer_length = len(answer)

                        if self.get_spot_info(start_loc) != self.P1 or \
                            next2[j][0] != self.HEIGHT - 1:
                            if self.get_spot_info(start_loc) != self.P2 or next2[j][0] != 0:

                                temp_move2 = [start_loc, next2[j]]

                                temp_board = Checkers(copy.deepcopy(self.board))
                                temp_board.step(self.move_to_action(temp_move2))

                                answer.extend(temp_board.get_capture_moves(temp_move2[1], temp_move1))

                        if len(answer) == answer_length:
                            answer.append(temp_move1)
        return answer


    def check_win_conditions(self):
      """
      Returns true when a player wins (other player has no pieces left)
      or draw.
      """

      # count player 1 and 2 pieces
      p1_pieces = 0
      p2_pieces = 0
      for i in range(self.HEIGHT):
        for j in range(self.WIDTH):
          if self.board[i, j] % 2 == 0 and self.board[i, j] > 0:
            p2_pieces += 1
          else:
            p1_pieces += 1

      # check for win
      if p2_pieces == 0:
        return 1  # player one wins

      if p1_pieces == 0:
        return -1  # player two wins

      return 0


    def get_piece_locations(self):
        """
        Gets all the pieces of the current player
        """
        piece_locations = []
        """
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if (self.player_turn == True and
                    (self.board[j][i] == self.P1 or self.board[j][i] == self.P1_K)) or \
                (self.player_turn == False and
                    (self.board[j][i] == self.P2 or self.board[j][i] == self.P2_K)):
                    piece_locations.append([j, i])
        """
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if self.board[j][i] == self.P1:
                    piece_locations.append([j, i])

        return piece_locations


    def legal_moves(self):
        """
        Gets the possible moves that can be made from the current board configuration.
        """
        actions = []
        piece_locations = self.get_piece_locations()
        capture_moves = list(reduce(lambda a, b: a + b, list(map(self.get_capture_moves, piece_locations))))

        if len(capture_moves) != 0:
            actions = np.zeros((len(capture_moves), 2))
            for i in range(len(capture_moves)):
                actions[i] = self.move_to_action(capture_moves[i])
            return actions.tolist()

        moves = list(reduce(lambda a, b: a + b, list(map(self.get_simple_moves, piece_locations))))
        actions = np.zeros((len(moves), 2))
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
        elif self.board[location[0]][location[1]] == self.P1:
            return self.P1_SYMBOL
        elif self.board[location[0]][location[1]] == self.P2:
            return self.P2_SYMBOL
        elif self.board[location[0]][location[1]] == self.P1_K:
            return self.P1_K_SYMBOL
        else:
            return self.P2_K_SYMBOL


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
