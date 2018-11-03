# python 3.5

import tensorflow as tf
import numpy as np


class NN():
	def __init__(self, lr, input_dim, num_hidden_layers, value_head_dim, policy_head_dim, filters, kernelsize, training, strides=1, padding="same"):
		""" 
		Args:
			lr (float): Learning rate
			input_dim (int tuple/list): Length, height, layers of input
			num_hidden_layers (int): Number of hidden layers
			filters (int): num features in output of convolution
			kernelsize (int tuple/list or int): Size of convolving window
			training (bool): True if model is training
			strides (int tuple/list or int): Stride of convolution
			padding ("same" or "valid"): "same" if 0 adding added during convolution
		"""
		self.lr = lr
		self.input_dim = input_dim
		self.num_hidden_layers = num_hidden_layers
		self.filters = filters
		self.kernelsize = kernelsize
		self.strides = strides
		self.padding = padding
		self.training = training

		self.inputs = tf.placeholder(shape=np.append(None,input_dim),dtype=tf.float32) # Variable batch size
		self.hidden_layers = _build_hidden_layers()
		self.value_head = _build_value_head()
		self.policy_head = _build_policy_head()

	def _build_hidden_layers(self):
		"""
		Returns:
			List of resBlocks
		"""
		hidden_layers = []
		resblk = resBlock(self.inputs, self.filters, self.kernelsize, True, self.strides, self.padding)
		hidden_layers.append(resblk)
		if num_hidden_layers > 1:
			for i in range(num_hidden_layers-1):
				resblk = resBlock(resblk, self.filters, self.kernelsize, True, self.strides, self.padding)
				hidden_layers.append(resblk)
		return hidden_layers

	def _build_value_head(self):
		vh = conv2d(self.hidden_layers[-1], (1,1), 1, "same", None)
		vh_bn = batch_norm(vh, self.training)
		vh_bn_relu = tf.nn.relu(vh_bn)
		vh_flat = tf.layers.flatten(vh_bn_relu)
		vh_dense = tf.layers.dense(
			inputs=vh_flat,
			units=20, # Arbitrary number. Consider decreasing for connect4.
			use_bias=False,
			activation=tf.nn.leaky_relu
		)
		vh_out = tf.layers.dense(
			inputs=vh_dense,
			units=1,
			use_bias=False,
			activation=tf.nn.tanh
		)

		return vh_out

    def _build_policy_head():
        return None

    def conv2d(inputs, filters, kernelsize, strides, padding, activation):
        return tf.layers.Conv2D(
            inputs=inputs,
            filters=filters,
            kernel_size=kernelsize,
            strides=strides,
            padding=padding,
            activation=activation
        )

	def batch_norm(inputs, training, BATCH_MOMENTUM = 0.997, BATCH_EPSILON = 1e-5):
		return tf.layers.batch_normalization(
			inputs=inputs, 
			axis=1, 
			momentum=_BATCH_NORM_DECAY, 
			epsilon=_BATCH_NORM_EPSILON, 
			center=True, 
			scale=True, 
			training=training, 
			fused=True)

    def resBlock(inputs, filters, kernelsize,
                 training, strides=1, padding="same"):
        """
        Args:
                inputs (tensor): Tensor input
                filter (int): Number of channels in the output
                kernelsize (int,tuple): Size of convolution window
                strides (int): Stride of convolution
                padding (int): "valid" or "same"
                training (bool): True if training
        """
        shortcut = tf.identity(inputs)
        conv1 = conv2d(inputs, filters, kernelsize, strides, padding, None)
        conv1_bn = batch_norm(conv1, training)
        conv1_bn_relu = tf.nn.relu(conv1_bn)
        conv2 = conv2d(conv1_bn_relu, filters,
                       kernelsize, strides, padding, None)
        conv2_bn = batch_norm(conv2, training)
        y = conv2_bn + shortcut
        y_relu = tf.nn.relu(y)
        return y