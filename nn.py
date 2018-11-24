# python 3.5

import tensorflow as tf
import numpy as np
import math


class NN():
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

    def create_directory(self,model_saver_path = './models/checkpoint/model.ckpt',
            final_model_saver_path='./model/checkpoint/model.ckpt', summary_path='./model/summary/'):
        """ create directories to store checkpoint files
        Args:
            model_saver_path: path for storing models obtained during training process
            final_model_saver_path: path for final mdoel
            summary_path: path for storing summaries of loss
        """
        globel_path = os.getcwd()
        globel_final_model_saver_path = os.path.join(globel_path, final_model_saver_path)
        global_model_saver_path = os.path.join(globel_path, model_saver_path)
        globel_summary_path = os.path.join(globel_path, summary_path)
        if os.path.exists(globel_final_model_saver_path):
            shutil.rmtree(globel_final_model_saver_path)
            os.mkdir(globel_final_model_saver_path)
        else:
            os.mkdir(globel_final_model_saver_path)

        if os.path.exists(global_model_saver_path):
            shutil.rmtree(global_model_saver_path)
            os.mkdir(global_model_saver_path)
        else:
            os.mkdir(global_model_saver_path)

        if os.path.exists(globel_summary_path):
            shutil.rmtree(globel_summary_path)
            os.mkdir(globel_summary_path)
        else:
            os.mkdir(globel_summary_path)

        return None

    def _build_hidden_layers(self):
        """
        Returns:
            List of resBlocks
        """
        hidden_layers = []

        # first convolutional layer is different
        initial_filter = tf.Variable(tf.random_uniform(
            [self.kernel_size, self.kernel_size, self.input_dim[2], self.filters]))
        first_layer = tf.nn.conv2d(
            self.inputs, initial_filter, self.strides, self.padding)
        hidden_layers.append(first_layer)

        resblk = self.resBlock(first_layer, self.filters,
                               True, strides=self.strides, padding=self.padding)
        hidden_layers.append(resblk)
        if self.num_hidden_layers > 1:
            for i in range(self.num_hidden_layers-1):
                resblk = self.resBlock(
                    resblk, self.filters, True, self.strides, self.padding)
                hidden_layers.append(resblk)
        return hidden_layers

    def _build_value_head(self):
        """
        Returns:
            vh_out (tf.dense, units=1): value estimation of current state
        """

        # goes back from n channels to 1
        vh_filter = tf.Variable(tf.random_uniform(
            [self.kernel_size, self.kernel_size, self.filters, 1]))
        vh = tf.nn.conv2d(
            self.hidden_layers[-1], vh_filter, [1, 1, 1, 1], "SAME")

        vh_bn = self.batch_norm(vh, self.training)
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

        # goes back from n channels to 1
        ph_filter = tf.Variable(tf.random_uniform(
            [self.kernel_size, self.kernel_size, self.filters, 1]))
        ph = tf.nn.conv2d(
            self.hidden_layers[-1], ph_filter, [1, 1, 1, 1], "SAME")

        ph_bn = self.batch_norm(ph, self.training)
        ph_bn_relu = tf.nn.relu(ph_bn)
        ph_flat = tf.layers.flatten(ph_bn_relu)
        ph_dense = tf.layers.dense(
            inputs=ph_flat,
            units=self.policy_head_dim,
            use_bias=False,
            activation=tf.nn.tanh,
            name='policy_head'
        )
        return ph_dense

    def conv2d(self, inputs, channels, strides, padding):
        return tf.nn.conv2d(
            input=inputs,
            filter=tf.Variable(tf.random_uniform(
                [self.kernel_size, self.kernel_size, channels, channels])),
            strides=strides,
            padding=padding,
        )

    def batch_norm(self, inputs, training, BATCH_MOMENTUM=0.997, BATCH_EPSILON=1e-5):
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
        shortcut = tf.identity(inputs)
        conv1 = self.conv2d(inputs, filters, strides, padding)
        conv1_bn = self.batch_norm(conv1, training)
        conv1_bn_relu = tf.nn.relu(conv1_bn)
        conv2 = self.conv2d(conv1_bn_relu, filters, strides, padding)
        conv2_bn = self.batch_norm(conv2, training)
        y = conv2_bn + shortcut
        y_relu = tf.nn.relu(y)
        return y

    def _cross_entropy_with_logits(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.policy_head, labels=self.policy_label)
        return tf.reduce_mean(loss)

    def _mean_sq_error(self):
        return tf.losses.mean_squared_error(self.value_label, self.value_head)

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

        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('mse_loss', self.mse_loss)
        tf.summary.scalar('total_loss', self.loss)

        if opt_type == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        apply_gradient_op = optimizer.minimize(self.loss)

        return apply_gradient_op


    def fit(self, X, v_lab, p_lab, batch_size = 100, epoch = 1000, model_saver_path = './model/checkpoint/model.ckpt',
            summary_path='./model/summary/'):
        """training model and save
        Args:
            X: input
            v_lab: value label
            p_lab: policy label
            batch_size: batch size for training data in every iteration
            epoch: training epochs
            model_saver_path: path for storing models obtained during training process
            summary_path: path for storing summaries of loss
        """

        train_iterations = math.ceil(X.shape[0]*epoch/batch_size)

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        saver = self.saver

        #if gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        #with tf.Session(config=config) as sess:

         # initialize session
        #self.pre_run(model_saver_path)
        #print("pre-run completed")

        with tf.Session() as sess:
            # Initialize session.
            sess.run(init)

            # Initialize summary writer.
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)


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
                if step % 1000 == 0:#store model every 1000 iteration times, may be changed due to # of network parameters
                    saver.save(sess, model_saver_path, global_step=step)

            saver.save(sess, model_saver_path)
        return None

    def pre_run(self, model_path='./model/checkpoint/model.ckpt'):

        meta_path = model_path+'.meta'

        # set the current session
        self.sess = tf.Session()
        #saver = tf.train.import_meta_graph(meta_path)
        self.saver.restore(self.sess, model_path)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

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

        with self.sess as sess:
            vh_pred, ph_pred = sess.run([self.value_head, self.policy_head], feed_dict={self.inputs: new_input, self.training: False})
            ph_pred = np.argmax(ph_pred, axis = 1)

        return [vh_pred, ph_pred]

