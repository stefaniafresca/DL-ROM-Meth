"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2019

"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import os

from Net import Net
import utils

seed = 374
np.random.seed(seed)

class DecNet(Net):
    def __init__(self, config):
        Net.__init__(self, config)

        self.n = config['n']
        self.n_params = config['n_params']

        self.size = 7
        self.n_layers = 3                                                         # hidden layers - 1
        self.n_neurons = 200
        self.n_h = config['n_h']

    def get_data(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, shape = [None, self.N_h])
            self.Y = tf.placeholder(tf.float32, shape = [None, self.n_params])

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()
            self.init = iterator.initializer

            self.output, self.params = iterator.get_next()

    def inference(self):
        # at testing time the encoder function is discarded
        fc_n = tf.layers.dense(self.params,
                               self.n_neurons,
                               activation = tf.nn.elu,
                               kernel_initializer = tf.keras.initializers.he_uniform())
        for i in range(self.n_layers):
            fc_n = tf.layers.dense(fc_n,
                                   self.n_neurons,
                                   activation = tf.nn.elu,
                                   kernel_initializer = tf.keras.initializers.he_uniform())
        u_n = tf.layers.dense(fc_n,
                              self.n,
                              activation = tf.nn.elu,
                              kernel_initializer = tf.keras.initializers.he_uniform())
        fc1_t = tf.layers.dense(u_n, 256, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc1_t')
        fc2_t = tf.layers.dense(fc1_t, self.N_h, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.he_uniform(), name = 'fc2_t')
        fc2_t = tf.reshape(fc2_t, [-1, self.n_h, self.n_h, 64])
        conv1_t = tf.layers.conv2d_transpose(inputs = fc2_t,
                                             filters = 64,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv1_t')
        conv2_t = tf.layers.conv2d_transpose(inputs = conv1_t,
                                             filters = 32,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv2_t')
        conv3_t = tf.layers.conv2d_transpose(inputs = conv2_t,
                                             filters = 16,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 2,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             activation = tf.nn.elu,
                                             name = 'conv3_t')
        conv4_t = tf.layers.conv2d_transpose(inputs = conv3_t,
                                             filters = 1,
                                             kernel_size = [self.size, self.size],
                                             padding = 'SAME',
                                             strides = 1,
                                             kernel_initializer = tf.keras.initializers.he_uniform(),
                                             name = 'conv4_t')
        feature_dim_dec = conv4_t.shape[1] * conv4_t.shape[2] * conv4_t.shape[3]
        self.u_h = tf.reshape(conv4_t, [-1, feature_dim_dec])

    def loss(self, u_h):
        with tf.name_scope('loss'):
            self.loss = self.omega_h * tf.reduce_mean(tf.reduce_sum(tf.pow(self.output - u_h, 2), axis = 1))

    def build(self):
        self.get_data()
        self.inference()
        self.loss(self.u_h)

    def test_once(self, sess, init):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_test, self.Y : self.params_test})
        total_loss = 0
        n_batches = 0
        self.U_h = np.zeros(self.S_test.shape)
        print('------------ TESTING ------------')
        try:
            while True:
                l, u_h = sess.run([self.loss, self.u_h])
                self.U_h[self.batch_size * n_batches : self.batch_size * (n_batches + 1)] = u_h
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss on testing set: {0}'.format(total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    #@profile
    def test_all(self):
        list = [v for v in tf.global_variables() if '_t' or 'dense' in v.name]
        saver = tf.train.Saver(var_list = list)

        if (self.large):
            S_train = utils.read_large_data(self.train_mat)
        else:
            S_train = utils.read_data(self.train_mat)
        idxs = np.random.permutation(S_train.shape[0])
        S_train = S_train[idxs]
        S_max, S_min = utils.max_min(S_train, self.n_train)
        del S_train

        print('Loading testing snapshot matrix...')
        if (self.large):
            self.S_test = utils.read_large_data(self.test_mat)
        else:
            self.S_test = utils.read_data(self.test_mat)

        utils.scaling(self.S_test, S_max, S_min)

        if (self.zero_padding):
            self.S_test = utils.zero_pad(self.S_test, self.p)

        print('Loading testing parameters...')
        self.params_test = utils.read_params(self.test_params)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                self.test_once(sess, self.init)

                utils.inverse_scaling(self.U_h, S_max, S_min)
                utils.inverse_scaling(self.S_test, S_max, S_min)
                n_test = self.S_test.shape[0] // self.N_t
                err = np.zeros((n_test, 1))
                for i in range(n_test):
                    num = np.sqrt(np.mean(np.linalg.norm(self.S_test[i * self.N_t : (i + 1) * self.N_t] - self.U_h[i * self.N_t : (i + 1) * self.N_t], 2, axis = 1) ** 2))
                    den = np.sqrt(np.mean(np.linalg.norm(self.S_test[i * self.N_t : (i + 1) * self.N_t], 2, axis = 1) ** 2))
                    err[i] = num / den
                print('Error indicator epsilon_rel: {0}'.format(np.mean(err)))
