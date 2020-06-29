"""

Stefania Fresca, MOX Laboratory, Politecnico di Milano
February 2019

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import scipy.io as sio
import h5py

def read_data(mat):
    data = sio.loadmat(mat)
    S = data['S'].squeeze()
    S = np.transpose(S)

    return S

def read_large_data(mat):
    file = h5py.File(mat, 'r')
    S = file['S'][:]

    return S

def read_params(mat):
    params = sio.loadmat(mat)
    params = params['I'].squeeze()

    return params

def max_min(S_train, n_train):
    S_max = np.max(np.max(S_train[:n_train], axis = 1), axis = 0)
    S_min = np.min(np.min(S_train[:n_train], axis = 1), axis = 0)

    return S_max, S_min

def scaling(S, S_max, S_min):
    S[ : ] = (S - S_min)/(S_max - S_min)

def inverse_scaling(S, S_max, S_min):
    S[ : ] = (S_max - S_min) * S + S_min

def zero_pad(S, n):
    paddings = np.zeros((S.shape[0], n))
    S = np.hstack((S, paddings))

    return S

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
