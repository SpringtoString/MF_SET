import numpy as np


def MPR(R_hat,R):
    '''

    :param R_hat: user dot item
    :param R: ratting
    :return: mean percentage rank
    '''
    rank = 0.0
    R_sum=np.sum(R)
    n_games=R.shape[1]
    R_hat_rank=np.argsort(np.argsort(-R_hat,axis=1))
    A=R*(R_hat_rank/n_games)
    rank = np.sum(A) / R_sum
    return rank