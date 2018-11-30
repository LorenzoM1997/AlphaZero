# keras implementation of nn.py

from keras import layers
from keras.engine.input_layer import Input
import numpy as np
import math
import os
import shutil

class NN():
	def __init__(self, input_dim, num_hidden_layers, policy_head_dim, lr=0.00025, kernel_size = 1, strides = 1):
		self.input_dim = input_dim
		return None

		self.model = _create_model()

	def _create_model(self):
		input_layer = Input(shape=self.input_dim, name='input_layer')

		x = Conv2D





def __init__(self, input_dim, num_hidden_layers, policy_head_dim, training, lr=0.00025, kernel_size = 3, filters=32, strides=1, padding="SAME"):
        """ 
        Args:
            input_dim (int tuple/list): Length, height, layers of input
            training (bool): True if model is training
            num_hidden_layers (int): Number of hidden layers
            lr (float): Learning rate
            filters (int): num features in output of convolution
            strides (int tuple/list or int): Stride of convolution
            padding ("SAME" or "valid"): "SAME" if 0 adding added during convolution
        """
        # make sure there is no other graph
        tf.reset_default_graph()

        self.lr = lr
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.filters = filters
        self.strides = [1, strides, strides, 1]
        self.padding = padding
        self.kernel_size = kernel_size
        self.policy_head_dim = policy_head_dim

        # Create directory, delete if exsited
        self.create_directory()

        self.inputs = tf.placeholder(tf.float32, shape=np.append(
            None, input_dim).tolist())  # Variable batch size
        self.training = tf.placeholder(tf.bool)
        self.policy_label = tf.placeholder(tf.float32,
                                           shape=np.append(None, policy_head_dim).tolist())
        self.value_label = tf.placeholder(tf.float32, [None, 1])
        self.hidden_layers = self._build_hidden_layers()
        self.value_head = self._build_value_head()
        self.policy_head = self._build_policy_head()
        self.ce_loss = self._cross_entropy_with_logits() # Why are these here?
        self.mse_loss = self._mean_sq_error() # ?
        self.train_op = self.train()
        self.saver = tf.train.Saver()