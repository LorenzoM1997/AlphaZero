import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, concatenate, add, Dropout, ReLU, Lambda, Activation, LeakyReLU

from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np

class NN():
    def __init__(self, input_shape, num_hidden_layers, policy_head_dim, lr=0.1, kernel_size=1, filters=32, strides=1, padding="same"):

        # parameters
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides

        # model
        self.input = Input(input_shape)
  
        self.c1 = self.conv_bn_rl(self.input, filters, kernel_size)
        self.maxp = MaxPool2D(3, strides=2, padding='same')(self.c1)
  
        self.res1 = self.resnet_block(self.maxp, filters, filters, 2)
        self.hidden_layers = []
        self.hidden_layers.append(self.res1)
        for i in range(num_hidden_layers - 1):
            x = self.resnet_block(self.hidden_layers[-1], filters, filters, 2)
            self.hidden_layers.append(x)
  
        self.flat = Flatten()(self.hidden_layers[-1])
  
        self.policy_head = Dense(policy_head_dim, activation='softmax')(self.flat)
        self.value_head = Dense(1, activation='tanh')(self.flat)

        self.model = Model(inputs = self.input, outputs = [self.policy_head, self.value_head])

        #loss
        self.model.compile(optimizer='sgd', loss = self.loss_function)

    def conv_bn_rl(self, x, f, k=1, s=1, p='same'):
        x = Conv2D(f, k, strides=s, padding=p)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
  
  
    def conv_block(self, tensor, f1, f2, s):
        x = self.conv_bn_rl(tensor, f1)
        x = self.conv_bn_rl(x, f1, 3, s=s)
        x = Conv2D(f2, 1)(x)
        x = BatchNormalization()(x)
  
        shortcut = Conv2D(f2, 1, strides=s, padding='same')(tensor)
        shortcut = BatchNormalization()(shortcut)
    
        x = add([shortcut, x])
        output = ReLU()(x)
    
        return output
  
  
    def identity_block(self, tensor, f1, f2):
        x = self.conv_bn_rl(tensor, f1)
        x = self.conv_bn_rl(x, f1, 3)
        x = Conv2D(f2, 1)(x)
        x = BatchNormalization()(x)
    
        x = add([tensor, x])
        output = ReLU()(x)
    
        return output
  
  
    def resnet_block(self, x, f1, f2, r):
        x = self.conv_block(x, f1, f2, self.strides)
    
        for _ in range(r-1):
            x = self.identity_block(x, f1, f2)
    
        return x
  

    def loss_function(self, yTrue, yPred):

        policy_label = K.softmax((yTrue[0] + 1) * 0.5) # normalize policy labels
        value_label = yTrue[1]
        policy_head = yPred[0]
        value_head = yPred[1]

        cross_entropy = keras.losses.categorical_crossentropy(policy_label, policy_head)
        mse = keras.losses.mean_squared_error(value_label, value_head)
        loss = K.sum(cross_entropy + mse)
        return loss

    #def pre_run(self, model_path='/model1/'):

    def fit(self, X, v_lab, p_lab, batch_size=100, epochs=1000, model_saver_path='/model1/'):
        p_lab = K.softmax((p_lab + 1) * 0.5)
        self.model.fit(X, [p_lab, v_lab], epochs=epochs, batch_size=batch_size)

    def pred(self, X):
        prediction = self.model.predict(X)
        print(prediction)
