"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
April 2019

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.stdout = open('*.out', 'w')

import utils
from ROMNet import ROMNet

if __name__ == '__main__':
    config = dict()
    config['n'] = 3
    config['n_params'] = 3
    config['lr'] = 0.0001
    config['omega_h'] = 0.5
    config['omega_n'] = 0.5
    config['batch_size'] = 40
    config['n_data'] = 44100
    config['N_h'] = 256
    config['n_h'] = 2
    config['N_t'] = 100
    config['train_mat'] = 'S_train.mat'
    config['test_mat'] = 'S_test.mat'
    config['train_params'] = 'params_train.mat'
    config['test_params'] = 'params_test.mat'
    config['checkpoints_folder'] = 'checkpoints'
    config['graph_folder'] = 'graphs'
    config['large'] = False                                                        # True if data are saved in .h5 format
    config['zero_padding'] = False                                                 # True if you must use zero padding
    config['p'] = 0                                                                # size of zero padding
    config['restart'] = False

    model = ROMNet(config)
    model.build()
    model.train_all(10000)
