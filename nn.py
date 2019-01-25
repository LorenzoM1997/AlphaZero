# python 3.5

import tensorflow as tf
import numpy as np
import math
import os
import shutil


class NN():
    def __init__(self, input_dim, num_convLayers, num_resBlocks, policy_head_dim, training, lr=0.00025, kernel_size=3, filters=32, strides=1, padding="same"):
        """ 
        Args:
            input_dim (int tuple/list): Length, height, layers of input
            training (bool): True if model is training
            num_resBlocks (int): Number of hidden layers
            lr (float): Learning rate
            filters (int): num features in output of convolution
            strides (int tuple/list or int): Stride of convolution
            padding ("SAME" or "valid"): "SAME" if 0 adding added during convolution
        """
        # make sure there is no other graph
        tf.reset_default_graph()

        self.lr = lr
        self.input_dim = input_dim
        self.num_convLayers = num_convLayers
        self.num_resBlocks = num_resBlocks
        self.filters = filters
        self.strides = (strides, strides)
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
        self.ce_loss = self._cross_entropy_with_logits()
        self.mse_loss = self._mean_sq_error()
        self.train_op = self.train()
        self.saver = tf.train.Saver()

    def create_directory(self, model_path='model'):
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

    def _build_hidden_layers(self):
        """
        Returns:
            List of resBlocks
        """

        # TODO: Create new scope Conv_Layers
        with tf.variable_scope('Residual_Blocks'):
            hidden_layers = []

            # first convolutional layer is different
            with tf.variable_scope('Residual_Block_1'):
                first_layer = self.conv2d(self.inputs)
                hidden_layers.append(first_layer)

                resblk = self.resBlock(first_layer, self.filters,
                                       True, strides=self.strides, padding=self.padding)
                hidden_layers.append(resblk)

            if self.num_resBlocks > 1:
                for i in range(self.num_resBlocks-1):
                    with tf.variable_scope('Residual_Block_'+str(i+2)):
                        resblk = self.resBlock(
                            resblk, self.filters, True, self.strides, self.padding)
                    
                    tf.summary.histogram('residual_block_'+str(i+2), resblk)

                    hidden_layers.append(resblk)

        return hidden_layers

    def _build_value_head(self):
        """
        Returns:
            vh_out (tf.dense, units=1): value estimation of current state
        """

        # goes back from n channels to 1
        with tf.variable_scope('Value_head'):
            vh = self.conv2d(self.hidden_layers[-1])
            vh_bn = self.batch_norm(vh, self.training)
            vh_bn_relu = tf.nn.relu(vh_bn)
            vh_flat = tf.layers.flatten(vh_bn_relu)
            vh_dense = tf.layers.dense(
                inputs=vh_flat,
                # Arbitrary number. Consider decreasing for connect4.
                units=10,
                use_bias=True,
                activation=tf.nn.leaky_relu
            )
            tf.summary.histogram('vh_dense', vh_dense)
            vh_out = tf.layers.dense(
                inputs=vh_dense,
                units=1,
                use_bias=True,
                activation=tf.nn.tanh
            )
            tf.summary.histogram('vh_out', vh_out)
        return vh_out

    def _build_policy_head(self):
        """
        Returns:
            ph_out (tf.dense, units=policy_head_dim): probability distribution 
        """
        with tf.variable_scope('Policy_head'):

            # goes back from n channels to 1
            ph = self.conv2d(self.hidden_layers[-1])
            ph_bn = self.batch_norm(ph, self.training)
            ph_bn_relu = tf.nn.relu(ph_bn)
            ph_flat = tf.layers.flatten(ph_bn_relu)
            ph_dense_1 = tf.layers.dense(
                inputs=ph_flat,
                units=20,
                use_bias=True,
                activation=tf.nn.leaky_relu
            )
            tf.summary.histogram('ph_dense_1', ph_dense_1)
            ph_out = tf.layers.dense(
                inputs=ph_dense_1,
                units=self.policy_head_dim,
                use_bias=True,
                activation=None
            )
            tf.summary.histogram('ph_out', ph_out)

        return ph_out

    def conv2d(self, inputs):
        return tf.layers.conv2d(
            inputs,
            self.filters,
            self.kernel_size,
            padding= self.padding,
            data_format = 'channels_first')

    def batch_norm(self, inputs, training, BATCH_MOMENTUM=0.9, BATCH_EPSILON=1e-5):
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=1,
            momentum=BATCH_MOMENTUM,
            epsilon=BATCH_EPSILON,
            center=True,
            scale=True,
            training=training,
            fused=True)

    def resBlock(self, inputs, filters, training, strides=1, padding="SAME"):
        """
        Args:
                inputs (tensor): Tensor input
                filter (int): Number of channels in the output
                strides (int): Stride of convolution
                padding (int): "valid" or "SAME"
                training (bool): True if training
        """
        shortcut = tf.nn.relu(tf.identity(inputs))
        conv1 = self.conv2d(inputs)
        conv1_bn = self.batch_norm(conv1, training)
        y = conv1_bn + shortcut
        return y

    def _cross_entropy_with_logits(self):
        with tf.variable_scope('Loss_in_policy_head'):
            normalized_policy = (1 + self.policy_label)/ 2
            tf.summary.histogram('policy_label', normalized_policy)
            loss = tf.losses.sigmoid_cross_entropy(normalized_policy,
                logits=self.policy_head)
        return loss

    def _mean_sq_error(self):
        with tf.variable_scope('Loss_in_value_head'):
            mse = tf.losses.mean_squared_error(
                self.value_label, self.value_head)
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

        if opt_type == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(self.lr, beta1 = 0.95)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        apply_gradient_op = optimizer.minimize(self.loss)

        return apply_gradient_op

    def fit(self, X, v_lab, p_lab, batch_size=100, epoch=1000, model_saver_path='/model1/'):
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
        # print(os.getcwd())
        model_saver_path = os.path.join(os.getcwd(), model_saver_path)
        # print(model_saver_path)
        if not os.path.exists(model_saver_path):
            os.mkdir(model_saver_path)
        train_iterations = math.ceil(X.shape[0]*epoch/batch_size)

        final_model_saver_path = os.path.join(model_saver_path, 'model.ckpt')
        model_saver_path = os.path.join(model_saver_path, 'model.ckpt')

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        saver = self.saver

        # if gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:

        with tf.Session() as sess:
            # Initialize session.
            sess.run(init)

            # Initialize summary writer.
            summary_writer = tf.summary.FileWriter(
                'model/summary', graph=sess.graph)

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

                if step % 1000 == 0:  # store model every 1000 iteration times, may be changed due to # of network parameters
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

    def pred(self, new_input):
        """
        args:
            new_input: a matrix of shape (1, num_layers, num_rows, num_cols)

        returns:
            a list [vh pred, ph_pred]
        """

        # if gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        vh_pred, ph_pred = self.sess.run([self.value_head, self.policy_head], feed_dict={
                                         self.inputs: new_input, self.training: False})

        return [vh_pred, ph_pred]
