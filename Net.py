"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2019

"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import os

import utils

seed = 374
np.random.seed(seed)

class Net:
    def __init__(self, config):
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.g_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

        self.n_data = config['n_data']
        self.n_train = int(0.8 * self.n_data)
        self.N_h = config['N_h']
        self.N_t = config['N_t']

        self.train_mat = config['train_mat']
        self.test_mat = config['test_mat']
        self.train_params = config['train_params']
        self.test_params = config['test_params']

        self.omega_h = config['omega_h']
        self.omega_n = config['omega_n']

        self.checkpoints_folder = config['checkpoints_folder']
        self.graph_folder = config['graph_folder']
        self.large = config['large']
        self.zero_padding = config['zero_padding']
        self.p = config['p']
        self.restart = config['restart']

    def get_data(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, shape = [None, self.N_h])
            self.Y = tf.placeholder(tf.float32, shape = [None, self.n_params])

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
            dataset = dataset.shuffle(self.n_data)
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()
            self.init = iterator.initializer

            input, self.params = iterator.get_next()
            self.input = tf.reshape(input, shape = [-1, int(np.sqrt(self.N_h)), int(np.sqrt(self.N_h)), 1])

    def inference(self):
        raise NotImplementedError("Must be overridden with proper definition of forward path")

    def loss(self, u_h, u_n):
        with tf.name_scope('loss'):
            output = tf.reshape(self.input, shape = [-1, self.N_h])
            self.loss_h = self.omega_h * tf.reduce_mean(tf.reduce_sum(tf.pow(output - u_h, 2), axis = 1))
            self.loss_n = self.omega_n * tf.reduce_mean(tf.reduce_sum(tf.pow(self.enc - u_n, 2), axis = 1))
            self.loss = self.loss_h + self.loss_n

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step = self.g_step)

    def summary(self):
        with tf.name_scope('summaries'):
            self.summary = tf.summary.scalar('loss', self.loss)

    def build(self):
        self.get_data()
        self.inference()
        self.loss(self.u_h, self.u_n)
        self.optimize()
        self.summary()

    def train_one_epoch(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_train, self.Y : self.params_train})
        total_loss_h = 0
        total_loss_n = 0
        total_loss = 0
        n_batches = 0
        print('------------ TRAINING -------------', flush = True)
        try:
            while True:
                _, l_h, l_n, l, summary = sess.run([self.opt, self.loss_h, self.loss_n, self.loss, self.summary])
                writer.add_summary(summary, global_step = step)
                step += 1
                total_loss_h += l_h
                total_loss_n += l_n
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss_h at epoch {0} on training set: {1}'.format(epoch, total_loss_h / n_batches))
        print('Average loss_n at epoch {0} on training set: {1}'.format(epoch, total_loss_n / n_batches))
        print('Average loss at epoch {0} on training set: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_val, self.Y : self.params_val})
        total_loss_h = 0
        total_loss_n = 0
        total_loss = 0
        n_batches = 0
        print('------------ VALIDATION ------------')
        try:
            while True:
                l_h, l_n, l, summary = sess.run([self.loss_h, self.loss_n, self.loss, self.summary])
                writer.add_summary(summary, global_step = step)
                total_loss_h += l_h
                total_loss_n += l_n
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        total_loss_mean = total_loss / n_batches
        if total_loss_mean < self.loss_best:
            saver.save(sess, self.checkpoints_folder + '/Net', step)
        print('Average loss_h at epoch {0} on validation set: {1}'.format(epoch, total_loss_h / n_batches))
        print('Average loss_n at epoch {0} on validation set: {1}'.format(epoch, total_loss_n / n_batches))
        print('Average loss at epoch {0} on validation set: {1}'.format(epoch, total_loss_mean))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return total_loss_mean

    def test_once(self, sess, init):
        start_time = time.time()
        sess.run(init, feed_dict = {self.X : self.S_test, self.Y : self.params_test})
        total_loss_h = 0
        total_loss_n = 0
        total_loss = 0
        n_batches = 0
        self.U_h = np.zeros(self.S_test.shape)
        print('------------ TESTING ------------')
        try:
            while True:
                l_h, l_n, l, u_h = sess.run([self.loss_h, self.loss_n, self.loss, self.u_h])
                self.U_h[self.batch_size * n_batches : self.batch_size * (n_batches + 1)] = u_h
                total_loss_h += l_h
                total_loss_n += l_n
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss_h on testing set: {0}'.format(total_loss_h / n_batches))
        print('Average loss_N on testing set: {0}'.format(total_loss_n / n_batches))
        print('Average loss on testing set: {0}'.format(total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    #@profile (if memory profiling must be used)
    def train_all(self, n_epochs):
        if (not self.restart):
            utils.safe_mkdir(self.checkpoints_folder)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('./' + self.graph_folder + '/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./' + self.graph_folder + '/test', tf.get_default_graph())

        print('Loading snapshot matrix...')
        if (self.large):
            S = utils.read_large_data(self.train_mat)
        else:
            S = utils.read_data(self.train_mat)

        idxs = np.random.permutation(S.shape[0])
        S = S[idxs]
        S_max, S_min = utils.max_min(S, self.n_train)
        utils.scaling(S, S_max, S_min)

        if (self.zero_padding):
            S = utils.zero_pad(S, self.p)

        self.S_train, self.S_val = S[:self.n_train, :], S[self.n_train:, :]
        del S

        print('Loading parameters...')
        params = utils.read_params(self.train_params)

        params = params[idxs]

        self.params_train, self.params_val = params[:self.n_train], params[self.n_train:]
        del params

        self.loss_best = 1
        count = 0
        with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
            sess.run(tf.global_variables_initializer())

            if (self.restart):
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.g_step.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, self.init, train_writer, epoch, step)
                total_loss_mean = self.eval_once(sess, saver, self.init, test_writer, epoch, step)
                if total_loss_mean < self.loss_best:
                    self.loss_best = total_loss_mean
                    count = 0
                else:
                    count += 1
                # early - stopping
                if count == 500:
                    print('Stopped training due to early-stopping cross-validation')
                    break
            print('Best loss on validation set: {0}'.format(self.loss_best))

        train_writer.close()
        test_writer.close()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoints_folder + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            print('Loading testing snapshot matrix...')
            if (self.large):
                self.S_test = utils.read_large_data(self.test_mat)
            else:
                self.S_test = utils.read_data(self.test_mat)

            utils.scaling(self.S_test, S_max, S_min)

            if (self.zero_padding):
                self.S_test = utils.zero_pad(self.S_test, self.n)

            print('Loading testing parameters...')
            self.params_test = utils.read_params(self.test_params)

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
