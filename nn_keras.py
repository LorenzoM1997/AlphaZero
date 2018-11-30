	# keras implementation of nn.py

from keras.layers import Conv2D, LeakyReLU
from keras.engine.input_layer import Input
import numpy as np
import math
import os
import shutil

class NN():
	def __init__(self, input_dim, num_hidden_layers, policy_head_dim, lr=0.00025, kernel_size = 1, strides = 1):
		self.input_dim = input_dim 
		self.model = self._create_model()

	def _conv2d(self, x, filters, kernel_size, activation, kernel_regularizer):
		x = Conv2D(
			filters=filters,
			kernel_size = kernel_size,
			activation = activation,
			kernel_regularizer = kernel_regularizer,
			data_format= 'channels_first',
			padding = 'same',
			use_bias= False
		)(x)

		return x

	def _create_model(self):
		input_layer = Input(shape=self.input_dim, name='input_layer')
		x = self._conv2d(input_layer, 8, 1, LeakyReLU(), None)
		return x

	def policy_head(self, x):

		x = Conv2D(
		filters = 2
		, kernel_size = (1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			self.output_dim
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'policy_head'
			)(x)

		return (x)

if __name__ == "__main__":
	nn = NN((3,3),2,8)


