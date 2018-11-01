# python 3.5

import tensorflow as tf
import numpy as np

class NN():
	""" 
	Args:
		lr (float): learning rate
		board_dim (int np_arr of size 2): length and width of board
		input_layers (int): number of input layers
		hidden_layers (int): number of hidden residual layers
	
	"""
	def __init__(self, lr, board_dim, input_layers, hidden_layers):
		self.lr = lr
		self.board_dim = board_dim
		self.input_layers = input_layers
		self.hidden_layers = hidden_layers


class resBlock(nn.Module):
	"""
	Args:
		inplanes (int): Number of channels in the input image
		filters (int): Number of channels in output image
		kernelsize (int): Size of convolving kernel
		padding (int): Padding added to sides of input
		stride (it): Stride of convolution
	"""
	def __init__(self, inplanes, filter, kernelsize, strides = 1, padding):
		input_layer = tf.reshape()
		self.conv1 = tf.layers.Conv2D(input_layer, kernel_size=kernelsize, filter=filter, strides=strides, padding=padding)