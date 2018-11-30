# keras implementation of nn.py

from keras import layers
import numpy as np
import math
import os
import shutil



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
