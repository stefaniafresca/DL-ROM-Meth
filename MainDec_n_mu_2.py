"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
February 2019

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.stdout = open('*.out', 'w')

import utils
from DecNet import DecNet

if __name__ == '__main__':
    config = dict()
    config['n'] = 3
    config['n_params'] = 3
    config['lr'] = 0.0001
    config['omega_h'] = 0.5
    config['omega_n'] = 0.5
    config['batch_size'] = 100
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
    config['large'] = False
    config['zero_padding'] = False
    config['p'] = 0
    config['restart'] = False

    model = DecNet(config)
    model.build()
    model.test_all()
