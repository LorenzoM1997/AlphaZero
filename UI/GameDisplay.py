from UI.TicTacToeUI import *
from UI.ConnectFourUI import *
from UI.CheckerUI import *

def DisplayMain(boardMatrix, gamename):
    totalMoves = len(boardMatrix)
    simulationsArray = []

    # parse input
    for i in range(0, totalMoves):
        simulationsArray.append(boardMatrix[i][0])

    rows = len(simulationsArray[0])
    cols = len(simulationsArray[0][0])

    # invert board for proper display
    inversions = int(totalMoves/2)
    for i in range(0, inversions):
        simulationsArray[2*i + 1] = (3 - simulationsArray[2*i + 1]) % 3

    if gamename == "TicTacToe":
        TicTacToe_Init(rows, cols, simulationsArray)
    elif gamename == "ConnectFour":
        ConnectFour_Init(rows, cols, simulationsArray)
    elif gamename == "Checkers":
        Checkers_Init(rows, cols, simulationsArray)
    else:
        print("game not recognized")
