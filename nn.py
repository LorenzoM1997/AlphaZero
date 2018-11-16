# python 3.5

import tensorflow as tf
import numpy as np
import math


class NN():
    def __init__(self, input_dim, num_hidden_layers, policy_head_dim, training, lr=0.001, filters=32, kernelsize=5, strides=1, padding="same", batch_size=128, train_steps=500):
        """ 
        Args:
            input_dim (int tuple/list): Length, height, layers of input
            training (bool): True if model is training
            num_hidden_layers (int): Number of hidden layers
            lr (float): Learning rate
            filters (int): num features in output of convolution
            kernelsize (int tuple/list or int): Size of convolving window
            strides (int tuple/list or int): Stride of convolution
            padding ("same" or "valid"): "same" if 0 adding added during convolution
            batch_size (int): Default 128, batch size in one training step
            train_steps (int): Default = 500, training iterations
        """
        self.lr = lr
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.filters = filters
        self.kernelsize = kernelsize
        self.strides = strides
        self.padding = padding
        self.training = training
        self.batch_size = batch_size

        self.inputs = tf.placeholder(tf.float32, shape=np.append(
            None, input_dim).tolist())  # Variable batch size
        self.policy_label = tf.placeholder(tf.float32,
                                           shape=np.append(None, policy_head_dim.shape).tolist())
        self.value_label = tf.placeholder(tf.float32, [None, 1])
        self.hidden_layers = self._build_hidden_layers()
        self.value_head = self._build_value_head()
        self.policy_head = self._build_policy_head()
        self.ce_loss = self._cross_entropy_with_logits()
        self.mse_loss = self._mean_sq_error()
        self.train_steps = train_steps

    def _build_hidden_layers(self):
        """
        Returns:
            List of resBlocks
        """
        hidden_layers = []
        resblk = self.resBlock(self.inputs, self.filters,
                               self.kernelsize, True, strides=self.strides, padding=self.padding)
        hidden_layers.append(resblk)
        if num_hidden_layers > 1:
            for i in range(num_hidden_layers-1):
                resblk = self.resBlock(
                    resblk, self.filters, self.kernelsize, True, self.strides, self.padding)
                hidden_layers.append(resblk)
        return hidden_layers

    def _build_value_head(self):
        """
        Returns:
            vh_out (tf.dense, units=1): value estimation of current state
        """
        vh = self.conv2d(self.hidden_layers[-1], (1, 1), 1, "same")
        vh_bn = batch_norm(vh, self.training)
        vh_bn_relu = tf.nn.relu(vh_bn)
        vh_flat = tf.layers.flatten(vh_bn_relu)
        vh_dense = tf.layers.dense(
            inputs=vh_flat,
            units=20,  # Arbitrary number. Consider decreasing for connect4.
            use_bias=False,
            activation=tf.nn.leaky_relu
        )
        vh_out = tf.layers.dense(
            inputs=vh_dense,
            units=1,
            use_bias=False,
            activation=tf.nn.tanh,
            name='value_head'
        )
        return vh_out

    def _build_policy_head(self):
        """
        Returns:
            ph_out (tf.dense, units=policy_head_dim): probability distribution 
        """
        ph = self.conv2d(self.hidden_layers[-1], (1, 1), 1, "same")
        ph_bn = batch_norm(ph, self.training)
        ph_bn_relu = tf.nn.relu(ph_bn)
        ph_flat = tf.layers.flatten(ph_bn_relu)
        ph_dense = tf.layers.dense(
            inputs=ph_dense,
            units=policy_head_dim,
            use_bias=False,
            activation=tf.nn.tanh,
            name='value_head'
        )
        return ph_dense

    def conv2d(self, inputs, filters, kernelsize, strides, padding):
        return tf.layers.Conv2D(
            inputs=inputs,
            filters=filters,
            kernel_size=kernelsize,
            strides=strides,
            padding=padding,
        )

    def batch_norm(self, inputs, training, BATCH_MOMENTUM=0.997, BATCH_EPSILON=1e-5):
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=1,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=training,
            fused=True)

    def resBlock(self, inputs, filters, kernelsize, training, strides=1, padding="same"):
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
        conv1 = self.conv2d(inputs, filters, kernelsize, strides, padding)
        conv1_bn = self.batch_norm(conv1, training)
        conv1_bn_relu = tf.nn.relu(conv1_bn)
        conv2 = self.conv2d(conv1_bn_relu, filters,
                            kernelsize, strides, padding)
        conv2_bn = batch_norm(conv2, training)
        y = conv2_bn + shortcut
        y_relu = tf.nn.relu(y)
        return y

    def _cross_entropy_with_logits(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.policy_head, labels=self.policy_label)
        return tf.reduce_mean(loss)

    def _mean_sq_error(self):
        return tf.losses.mean_squared_error(self.value_label, self.value_head)

    def getBatch(self, X, train_steps, batch_size, value_labels, policy_labels):
        """
        args:
            value_labels
            policy_labels
        """
        sample_size = X.shape[0]
        startIndex = (train_steps * batch_size) % sample_size
        endIndex = startIndex + batch_size % sample_size
        if startIndex < endIndex:
            batch_X = X[startIndex: endIndex]
            batch_Y = value_labels[startIndex: endIndex]
            batch_Z = policy_labels[startIndex: endIndex]
        else:
            batch_X_1 = X[startIndex:]
            batch_X_2 = X[:endIndex]
            batch_X = np.concatenate((batch_X_1, batch_X_2), axis=0)
            batch_Y_1 = valueLabels[startIndex:]
            batch_Y_2 = valueLabels[:endIndex]
            batch_Y = np.concatenate((batch_Y_1, batch_Y_2), axis=0)
            batch_Z_1 = policy_labels[startIndex:]
            batch_Z_2 = policy_labels[:endIndex]
            batch_Z = np.concatenate((batch_Z_1, batch_Z_2), axis=0)

        return batch_X, batch_Y, batch_Z

    def fit(self, X, v_lab, p_lab, batch_size,
            opimizer='AdamOptimizer', saver_path='./model/checkpoint/model.ckpt'):
        """
        Args:
            X: input
            v_lab: value label
            p_lab: policy label
        """
        init = tf.global_variables_initializer()

        self.loss = self.ce_loss + self.mse_loss

        if optimizer == 'AdamOptimizer':
            train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        if optimizer == 'GradientDescentOptimizer':
            train_step = tf.train.GradientDescentOptimizer(
                lr).minimize(self.loss)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in range(self.train_steps):
                [batch_X, batch_Y, batch_Z] = getBatch(
                    X, step, self.batch_size, value_labels, policy_labels)
                train_step.run(feed_dict={X: batch_X, Y: batch_Y, Z: batich_Z})

            saved_path = saver.save(sess, saver_path)
        return None

    def pred(new_input, saver_path='./model/checkpoint/model.ckpt'):
        meta_path = saver_path+'.meta'
        model_path = saver_path
        saver = tf.train.import_meta_graph(meta_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            graph = tf.get_default_graph()
            value_prob_op = graph.get_operation_by_name('value_head')
            value_pred = graph.get_tensor_by_name('value_head:0')
            vh_pred = sess.run(value_pred, feed_dict={X: new_input})
            policy_prob_op = graph.get_operation_by_name('policy_head')
            policy_pred = graph.get_tensor_by_name('policy_head:0')
            ph_pred = sess.run(policy_pred, feed_dict={X: new_input})
            ph_pred = tf.argmax(ph_pred, axis=1)
            pred = [vh_pred, ph_pred]
        return pred
