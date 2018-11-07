import numpy as np
from copy import deepcopy

def get_set_generic(name):
    return property(lambda self: getattr(self.game, name),
                    lambda self, val: setattr(self.game, name, val))

class GameGlue:

    num_rows = get_set_generic('num_rows') 
    num_cols = get_set_generic('num_cols') 
    num_layers = get_set_generic('num_layers') 
    action_space = get_set_generic('action_space') 
    obs_space = get_set_generic('obs_space') 
    terminal = get_set_generic('terminal') 
    name = get_set_generic('name') 
    board = get_set_generic('board')

    def __init__(self, game):
        self.game = game
        self.last_state = None
        self.ended = 0 

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        self.last_state = state[0]
        self.ended = state[1]
        board = []
        for i in state[2]:
            board.append(np.array(i, dtype=np.uint8))
        self.game.board = np.array(board, dtype=np.uint8)

    @state.getter
    def state(self):
        board = (),
        for i in self.game.board:
            board = board + (tuple(i),)
        return (self.last_state, self.ended, board[1:])

    def restart(self):
        self.game.restart()
        self.last_state = None
        self.ended = 0

    def is_valid(self, action):
        return self.game.is_valid(action)

    def legal_moves(self):
        return self.game.legal_moves()

    def invert_board(self):
        self.game.invert_board()

    def step(self, action):
        self.last_state = self.game.step(action)
        self.ended = self.game.terminal
        return self.last_state
    def render(self):
        self.game.render()

    def starting_state(self):
        return self.__class__(self.game.__class__()).state

    def pack_state(self, data):
        self.state = data
        return self.state

    def unpack_state(self, state):
        return self.state

    def next_state(self, state, action):
        game_copy = deepcopy(self)
        game_copy.state = state
        game_copy.step(action)
        game_copy.invert_board()
        return game_copy.state

    def is_legal(self, history, action):
        game_copy = deepcopy(self)
        game_copy.state = history[-1]
        return action in game_copy.legal_moves()

    def legal_actions(self, history):
        game_copy = deepcopy(self)
        game_copy.state = history[-1]
        return game_copy.legal_moves()

    def is_ended(self, history):
        game_copy = deepcopy(self)
        game_copy.state = history[-1]
        return game_copy.ended

    def win_values(self, history):
        game_copy = deepcopy(self)
        game_copy.state = history[-1]

        if not game_copy.ended:
            return

        if game_copy.last_state == 1:
            return 1
        elif game_copy.last_state == 0:
            return 0
        else:
            return -1

