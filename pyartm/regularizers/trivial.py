import numpy as np


class Regularizer(object):
    def __call__(self, phi, theta, n_tw, n_dt , n_dw = 0):
        return np.zeros_like(n_tw), np.zeros_like(n_dt)
