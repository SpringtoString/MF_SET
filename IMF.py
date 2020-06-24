import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
import time
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn



class IMF(nn.Module):
    def __init__(self, n_user, n_item, n_factor):
        '''
        :param n_user:
        :param n_item:
        :param n_factor:
        '''
        super(IMF, self).__init__()
        self.n_factor = n_factor
        self.n_user = n_user
        self.n_item = n_item
        self.X = Parameter(torch.randn([self.n_user, self.n_factor], dtype=torch.double))
        self.Y = Parameter(torch.randn([self.n_item, self.n_factor], dtype=torch.double))

    def forward(self):
        A = torch.mm(self.X, self.Y.T)
        return A

def MPR(R_hat,R):
    '''
    :param R_hat: user dot item
    :param R: ratting
    :return: mean percentage rank
    '''
    R_sum=torch.sum(R)
    n_item=R.size()[1]
    R_hat_rank=torch.argsort(torch.argsort(-R_hat,dim=1))
    A=R*(R_hat_rank/torch.tensor(n_item , dtype=torch.double))
    rank = torch.sum(A) / R_sum
    return rank.item()

if __name__ == '__main__':
    train_data = pd.read_table('data1.txt', header=None)
    test_data = pd.read_table('test1.txt', header=None)
    n_users = len(set(train_data[0].tolist() + test_data[0].tolist()))
    n_items = len(set(train_data[1].tolist() + test_data[1].tolist()))
    print('There are {0} users and {1} games in the data'.format(n_users, n_items))

    user2idx = {user: i for i, user in enumerate(set(train_data[0].tolist() + test_data[0].tolist()))}
    idx2user = {i: user for user, i in user2idx.items()}

    item2idx = {game: i for i, game in enumerate(set(train_data[1].tolist() + test_data[1].tolist()))}
    idx2item = {i: game for game, i in item2idx.items()}

    user_train_idx = train_data[0].apply(lambda x: user2idx[x]).values
    item_train_idx = train_data[1].apply(lambda x: item2idx[x]).values
    ratting_train = np.array([1 for i in range(len(train_data))])

    user_test_idx = test_data[0].apply(lambda x: user2idx[x]).values
    item_test_idx = test_data[1].apply(lambda x: item2idx[x]).values
    ratting_test = np.array([1 for i in range(len(test_data))])

    zero_matrix = np.zeros(shape=(n_users, n_items))  # Create a zero matrix

    R = zero_matrix.copy()
    R[user_train_idx, item_train_idx] = ratting_train

    P = zero_matrix.copy()
    P[user_train_idx, item_train_idx] = 1  # Fill the matrix will preferences (bought)                Pui

    C = zero_matrix.copy()
    # Fill the confidence with (hours played)
    # Added 1 to the hours played so that we have min. confidence for games bought but not played.    Cui
    alpha = 20
    C[user_train_idx, item_train_idx] = alpha * ratting_train + 1

    R_test = zero_matrix.copy()
    R_test[user_test_idx, item_test_idx] = ratting_test

    ###================处理数据完成===================###

    C = torch.tensor(C, dtype=torch.double)
    R = torch.tensor(R, dtype=torch.double)
    R_test[user_test_idx, item_test_idx] = ratting_test

    model=IMF(n_users, n_items, n_factor=10)
    learning_rate=0.01
    optimizer=torch.optim.Adam(params = model.parameters() , lr = learning_rate)

    for epoch in range(200):
        optimizer.zero_grad()
        R_hat = model()
        loss = C * ((R_hat - R).pow(2))
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        if (epoch % 10 == 0):
            with torch.no_grad():
                print( epoch, "mpr is ", MPR(R_hat, R_test) )