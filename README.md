# AlphaZero
ECS 171 Machine Learning project

## Setup

Everyone should have installed

    pip install tensorflow
    pip install numpy
    pip install autopep8
    
## Submission

Before submitting a file, format the file by running

    autopep8 --in-place filename.py

## Game implementation
The following functions need to be implemented for each game:

    restart(self)
    
    is_valid(self, action)
        # returns a boolean
    
    invert_board(self)
    
    step(self, action)
        # returns reward
        
    render(self)
    
    
