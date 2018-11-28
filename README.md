# AlphaZero
ECS 171 Machine Learning project

## Setup

Some packages require an older version of python installed (particularly tensorflow, which as of this writing requires python3.5). To set up a virtual environment with the proper version, you should first have virtualenv installed, and then run:

    virtualenv -p python3.5 venv
    source venv/bin/activate

Everyone should have installed:

    pip install autopep8
    pip install progressbar2

You need to have a Nvidia gpu, you should try to install the gpu version of tensorflow

    pip install tensorflow-gpu
and all the related requirements, such as Cuda and CuDnn.

Before running the program, create an empty folder named 'saved' if not present.

## Running

There are different modes that can be used. For training mode do:

python3 training.py
python3 main.py

for all other moves, you need to change the mode in the code at line 150 to either 'evaluation', 'debug' or 'manual'
