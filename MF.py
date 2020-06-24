'''

'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
import time
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn



class MF(nn.Module):
    def __init__(self, n_user, n_item,  n_factor):
       '''
       :param C:
       :param R_test:
       :param n_factor:
       '''
       super(MF, self).__init__()
       self.n_factor = n_factor
       self.n_user = n_user
       self.n_item = n_item
       self.X = Parameter(torch.randn([self.n_user, self.n_factor], dtype=torch.double))
       self.Y = Parameter(torch.randn([self.n_item, self.n_factor], dtype=torch.double))

    def forward(self):
        A=torch.mm(self.X , self.Y.T)
        return A

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

    user_idx = np.concatenate([train_data[0].apply(lambda x: user2idx[x]).values , test_data[0].apply(lambda x: user2idx[x]).values])
    item_idx = np.concatenate([train_data[1].apply(lambda x: item2idx[x]).values , test_data[1].apply(lambda x: item2idx[x]).values])
    ratting = np.concatenate([train_data[2].values , test_data[2].values])

    having= np.array([1 for i in range(len(train_data)+len(test_data))])

    zero_matrix = np.zeros(shape=(n_users, n_items))  # Create a zero matrix
    R = zero_matrix.copy()
    R[user_idx, item_idx] = ratting

    I = zero_matrix.copy()
    I[user_idx, item_idx] = having
    ###================处理数据完毕=====================###

    I = torch.tensor(I,dtype=torch.double)
    R = torch.tensor(R, dtype=torch.double)
    model = MF(n_users, n_items, n_factor=10)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(200):
        R_hat = model()
        loss=I*((R_hat-R).pow(2))
        loss=loss.mean()
        if (epoch % 10 == 0):
            print(epoch, "loss is " , loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()