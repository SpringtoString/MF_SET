import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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