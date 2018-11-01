#!/usr/bin/env python
import sys
from pkg_resources import iter_entry_points
import server

import new_board


args = sys.argv[1:]
addr, port = None, None

board = new_board.Board

if len(args) > 1:
    addr = args[1]
if len(args) > 2:
    port = int(args[2])


api = server.Server(board(), addr, port)
api.run()
