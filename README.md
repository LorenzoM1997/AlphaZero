# AlphaZero
ECS 171 Machine Learning project

## Setup

To run the program correctly you need to install:

    pip install autopep8
    pip install progressbar2

You need to have a Nvidia gpu, you can install the gpu version of tensorflow

    pip install tensorflow-gpu
and all the related requirements, such as Cuda and CuDnn.

Before running the program, create an empty folder named 'saved' if not already there.

## Running

There are different modes that can be used. For training mode do:

    python3 training.py
    python3 main.py

for all other moves, you need to change the mode in the code at line 193 to either 'evaluation', 'debug' or 'manual'
