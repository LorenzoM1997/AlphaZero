# python 3.5

import tensorflow as tf
import numpy as np
import math
import os
import shutil
class NN():
    def __init__(self, input_dim, policy_head_dim, training, channels = 3, lr=0.00025, kernel_size = 1, filters=32, strides=1, padding="SAME"):
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
        # self.num_hidden_layers = num_hidden_layers
        self.channels = channels
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
        self.conv1 = self._conv1()
        self.conv2 = self._conv2()
        self.value_head = self._build_value_head()
        self.policy_head = self._build_policy_head()
        self.ce_loss = self._cross_entropy_with_logits()
        self.mse_loss = self._mean_sq_error()
        self.train_op = self.train()
        self.saver = tf.train.Saver()

    def create_directory(self,model_path = 'model'):
        """ create directories to store checkpoint files
        Args:
            model_saver_path: path for storing model obtained during training process
            final_model_saver_path: path for final mdoel
            summary_path: path for storing summaries of loss
        """

        # Create parent directory
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Create directory for all checkpoints and summary
        model_saver_path = model_path+'/checkpoint'

        if not os.path.exists(model_saver_path):
            os.mkdir(model_saver_path)
        return None

    def _conv1(self):
        filter_tensor=tf.Variable(tf.random_uniform(
                [self.kernel_size, self.kernel_size, self.channels, self.channels]))
        conv = tf.nn.conv2d(input=self.inputs, filter = filter_tensor, strides=self.strides, padding=self.padding)
        return conv

    def _conv2(self):
        filter_tensor=tf.Variable(tf.random_uniform(
                [self.kernel_size, self.kernel_size, self.channels, self.channels]))
        conv = tf.nn.conv2d(input=self.conv1, filter = filter_tensor, strides=self.strides, padding=self.padding)
        return conv

    def _build_value_head(self):
        """
        Returns:
            vh_out (tf.dense, units=1): value estimation of current state
        """

        # goes back from n channels to 1
        with tf.variable_scope('Value_head'):
            vh_bn_relu = tf.nn.relu(self.conv2)
            vh_flat = tf.layers.flatten(vh_bn_relu)
            vh_out = tf.layers.dense(
                inputs=vh_flat,
                units=1,  # Arbitrary number. Consider decreasing for connect4.
                use_bias=False,
                activation=tf.nn.leaky_relu
            )
        return vh_out

    def _build_policy_head(self):
        """
        Returns:
            ph_out (tf.dense, units=policy_head_dim): probability distribution 
        """
        with tf.variable_scope('Policy_head'):

            # goes back from n channels to 1
            ph_bn_relu = tf.nn.relu(self.conv2)
            ph_flat = tf.layers.flatten(ph_bn_relu)
            ph_out = tf.layers.dense(
                inputs=ph_flat,
                units=self.policy_head_dim,
                use_bias=False,
                activation=tf.nn.tanh
            )

        return ph_out

    def _cross_entropy_with_logits(self):
        with tf.variable_scope('Loss_in_policy_head'):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.sigmoid(self.policy_label),
                logits=self.policy_head)
            loss = tf.reduce_mean(loss)
        return loss

    def _mean_sq_error(self):
        with tf.variable_scope('Loss_in_value_head'):
            mse = tf.losses.mean_squared_error(self.value_label, self.value_head)
        return mse

    def getBatch(self, X, train_step, batch_size, value_labels, policy_labels):
        """ Set batch size for each training step
        Args:
            value_labels
            policy_labels
        Return:
            batch_X: predictors in assigned batch size
            batch_Y: value labels in training
            batch_Z: policy labels in training
        """
        sample_size = X.shape[0]
        startIndex = (train_step * batch_size) % sample_size
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

    def train(self, opt_type='AdamOptimizer'):
        """declare optimal method and add training layer to graph
        Args:
            opt_type:optimizing algorithm, sgd or adam
        Return:
            training graph
        """
        self.loss = self.ce_loss + self.mse_loss

        tf.summary.scalar('policy_head_loss', self.ce_loss)
        tf.summary.scalar('value_head_loss', self.mse_loss)
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.histogram('ph', self.policy_head)
        tf.summary.histogram('convolution_layer2', self.conv2)

        if opt_type == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        apply_gradient_op = optimizer.minimize(self.loss)

        return apply_gradient_op


    def fit(self, X, v_lab, p_lab, batch_size = 100, epoch = 1000, model_saver_path = '/model1/'):
        """training model and save
        Args:
            X: input
            v_lab: value label
            p_lab: policy label
            batch_size: batch size for training data in every iteration
            epoch: training epochs
            model_saver_path: path for storing model obtained during training process
            summary_path: path for storing summaries of loss
        """
        if not os.path.exists(model_saver_path):
            os.mkdir(model_saver_path)
        train_iterations = math.ceil(X.shape[0]*epoch/batch_size)

        model_saver_path = os.getcwd() + model_saver_path
        final_model_saver_path = model_saver_path + 'model.ckpt'
        model_saver_path += 'model.ckpt'

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        saver = self.saver

        #if gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        #with tf.Session(config=config) as sess:


        with tf.Session() as sess:
            # Initialize session.
            sess.run(init)

            # Initialize summary writer.
            summary_writer = tf.summary.FileWriter('model/summary', graph=sess.graph)

            for step in range(train_iterations):
                [batch_X, batch_Y, batch_Z] = self.getBatch(
                    X, step, batch_size, v_lab, p_lab)

                feed_dict = {self.inputs: batch_X, 
                             self.value_label: batch_Y, 
                             self.policy_label: batch_Z, 
                             self.training: True}

                sess.run(self.train_op, feed_dict=feed_dict)

                if step % 20 == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                if step % 1000 == 0: #store model every 1000 iteration times, may be changed due to # of network parameters
                    saver.save(sess, model_saver_path, global_step=step)

            saver.save(sess, final_model_saver_path)
        return None

    def pre_run(self, model_path='/model1/'):

        model_saver_path = os.getcwd() + model_path
        model_path += 'model.ckpt'
        meta_path = model_path+'.meta'

        # set the current session
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(meta_path)
        self.saver.restore(self.sess, model_path)

    def pred(self,new_input):
        """
        args:
            new_input: a matrix of shape (1, num_layers, num_rows, num_cols)

        returns:
            a list [vh pred, ph_pred]
        """

        #if gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        #with tf.Session(config=config) as sess:
        vh_pred, ph_pred = self.sess.run([self.value_head, self.policy_head], feed_dict={self.inputs: new_input, self.training: False})

        return [vh_pred, ph_pred]

