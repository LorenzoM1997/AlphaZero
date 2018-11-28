from __future__ import division
import numpy as np
import time
from math import log, sqrt
from random import choice


class Stat(object):
    __slots__ = ('value', 'visits')

    def __init__(self, value=0, visits=0):
        self.value = value
        self.visits = visits


class UCT(object):
    def __init__(self, board, **kwargs):
        self.board = board
        self.history = []
        self.stats = {}

        self.use_nn = False
        self.name = 'new'

        self.max_depth = 0
        self.data = {}
        self.DEBUG = True
        self.memorize = True

        # Heauristics
        self.referenceTime = time.time()
        self.useDiscounting = False

        self.calculation_time = float(kwargs.get('time', 10))
        self.max_actions = int(kwargs.get('max_actions', 1000))

        # Exploration constant, increase for more exploratory actions,
        # decrease to prefer actions with known higher win rates.
        self.C = float(kwargs.get('C', 1.4))

    def update(self, state):
        self.history.append(self.board.pack_state(state))

    def get_action(self, names, inputs, outputs):
        # Causes the AI to calculate the best action from the
        # current game state and return it.

        self.max_depth = 0
        self.data = {}
        self.stats.clear()
        if self.memorize:
            try:
                current_stats = self.stats[self.history[-1]]
                games = current_stats.visits
            except:
                games = 0
        else:
            self.stats.clear()

        state = self.history[-1]
        legal = self.board.legal_actions(state)

        # Bail out early if there is no real choice to be made.
        if not legal:
            return
        if len(legal) == 1:
            return legal[0]

        begin = time.time()
        self.referenceTime = time.time()
        while time.time() - begin < self.calculation_time:
            if games >= 1600:
                break
            self.run_simulation(names, inputs, outputs)
            games += 1  

        names.put('done')

        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        self.data.update(games=games, max_depth=self.max_depth,
                         time=str(time.time() - begin))

        # Store and display the stats for each possible action.
        self.data['actions'] = self.calculate_action_values(state, legal)
        if self.DEBUG:
            print(self.data['games'], self.data['time'])
            print("Maximum depth searched:", self.max_depth)
            for m in self.data['actions']:
                print(self.action_template.format(**m))

        # return mcts policy
        self.policy = np.zeros(len(self.board.action_space))
        for m in self.data['actions']:
            self.policy[m['action']] = m['average']

        # Pick the action with the highest average value.
        return self.data['actions'][0]['action']

    def run_simulation(self, names, inputs, outputs):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.

        # A bit of an optimization here, so we have a local
        # variable lookup instead of an attribute access each loop.
        stats = self.stats
        visited_states = []
        history_copy = self.history[:]
        state = history_copy[-1]

        expand = True
        for t in range(1, self.max_actions + 1):
            legal = self.board.legal_actions(history_copy[-1])
            actions_states = [(p, self.board.next_state(state, p))
                              for p in legal]

            if all(S in stats for p, S in actions_states):
                # If we have stats on all of the legal actions here, use UCB1.
                log_total = log(
                    sum(stats[S].visits for p, S in actions_states) or 1)
                value, action, state = max(
                    ((stats[S].value / (stats[S].visits or 1)) +
                     self.C * sqrt(log_total / (stats[S].visits or 1)), p, S)
                    for p, S in actions_states
                )
            else:
                # Otherwise, just make an arbitrary decision.
                if self.use_nn:
                    names.put(self.name)
                    inputs.put([self.board.game.layers()])
                    vh_pred, ph_pred = outputs.get()
                    ph_pred = ph_pred[0]
                    for p, state in actions_states:
                        stats[state] = Stat()
                        stats[state].value = ph_pred[p]
                        stats[state].visits = 1
                    value, action, state = max(
                        ((stats[S].value / (stats[S].visits or 1)) , p, S)
                        for p, S in actions_states)
                else:
                    action, state = choice(actions_states)

            history_copy.append(state)

            if expand and state not in stats:
                expand = False
                stats[state] = Stat()
                if t > self.max_depth:
                    self.max_depth = t

            visited_states.append(state)

            # check if the game is ended
            if history_copy[-1][1]:
                break

        # Back-propagation
        end_values = self.end_values(history_copy[-1])
        multiplier = -1
        for i in range(len(visited_states)):
            multiplier *= -1
            state = visited_states[len(visited_states) - 1 - i]
            if state not in stats:
                continue
            S = stats[state]
            S.visits += 1

            if self.useDiscounting:
                S.value += (end_values * multiplier)/self.referenceTime
            else:
                S.value += end_values * multiplier


class UCTValues(UCT):
    action_template = "{action}: {average:.1f} ({sum} / {plays})"

    def __init__(self, board, **kwargs):
        super(UCTValues, self).__init__(board, **kwargs)
        self.end_values = board.win_values

    def calculate_action_values(self, state, legal):
        actions_states = ((p, self.board.next_state(state, p)) for p in legal)
        return sorted(
            ({'action': p,
              'average': self.stats[S].value / self.stats[S].visits,
              'sum': self.stats[S].value,
              'plays': self.stats[S].visits}
             for p, S in actions_states),
            key=lambda x: (x['average'], x['plays']),
            reverse=True
        )

