from training import NetTrainer
from training import training_nn

def init(game_interface):
    global Trainer
    Trainer = NetTrainer(game_interface)
