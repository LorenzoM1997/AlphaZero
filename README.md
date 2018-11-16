# AlphaZero
ECS 171 Machine Learning project

## Setup

Some packages require an older version of python installed (particularly tensorflow, which as of this writing requires python3.5). To set up a virtual environment with the proper version, you should first have virtualenv installed, and then run:

    virtualenv -p python3.5 venv
    source venv/bin/activate

Everyone should have installed:

    pip install tensorflow
    pip install numpy
    pip install autopep8
    pip install progressbar2

If you have a Nvidia gpu, you should try to install the gpu version of tensorflow

    pip install tensorflow-gpu
and all the related requirements, such as Cuda and CuDnn.

## Uploads

Before uploading a file, format the file by running

    autopep8 --in-place filename.py
## Monte Carlo Tree Search
The Monte Carlo Tree search is implemented in the file `uct.py`. It needs GameGlue() to work correcly.
The improvements now needed in the file are:
1. save the Monte Carlo tree
2. load the Monte Carlo tree

## Asychronous behavior
When running `main.py` the file uses the multiprocessing module to run different processes in parallel. This reduces the time needed to run the episodes and allow us to use all the CPU resources. Without training the neural network, the program seems to be able to run without any problem 3 simulations at a time with laptop CPU (probably up to 5).

The games simulated are saved in a single file at each session.

## Memory

The episodes are saved with pickle, to open it you need to import it

    import pickle

The episodes for a game are saved in a file, which filename is the game name + a timestamp. You can know the exact name by looking at the variable name of the class Game() in Games.py. You can extract the data with:

    data = pickle.load(open(filename, "rb"))
    
The file contains a list of lists.
Each entry in the list is therefore a list with 3 elements
1. A representation of the board
2. The action performed
3. The reward associated to the game for that particular player.

## Game implementation
All the games are saved in standalone files in the folder Games. In the folder there is also a file, Games.py where the class Game is implemented. There are currently two games available
1. TicTacToe()
2. ConnectFour()

The following methods need to be implemented for each game:

    invert_board()
    is_valid(action): returns a boolean
    layers(): returns a np.ndarray with one-hot encoded info
    legal moves() : returns a list
    render()
    restart()
    step(action): returns reward
    
The games should also have variables such as

    action_space
    name
    num_cols
    num_layers
    num_rows
    observation_space
    terminal
