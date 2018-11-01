#!/usr/bin/env python
import sys
import argparse

from pkg_resources import iter_entry_points

import new_board
import player
import uct


board_plugins = dict(
    (ep.name, ep.load())
    for ep in iter_entry_points('jrb_board.games')
)

player_plugins = dict(
    (ep.name, ep.load())
    for ep in iter_entry_points('jrb_board.players')
)

parser = argparse.ArgumentParser(
    description="Play a boardgame using a specified player type.")
parser.add_argument('game')
parser.add_argument('player')
parser.add_argument('address', nargs='?')
parser.add_argument('port', nargs='?', type=int)
parser.add_argument('-e', '--extra', action='append')

args = parser.parse_args()

player_obj = None

if args.player == 'human':
    player_obj = player.HumanPlayer
elif args.player == 'bot':
    player_obj = uct.UCTValues

board = new_board.Board
player_kwargs = dict(arg.split('=') for arg in args.extra or ())


client = player.Client(player_obj(board(), **player_kwargs),
                       args.address, args.port)
client.run()
