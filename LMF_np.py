import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
import time

class LogisticMF():

    def __init__(self, counts,R, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=200):
        self.counts = counts
        self.R=R
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model(self):

        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        #梯度累计和
        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))

        for i in range(self.iterations):
            t0 = time.time()
            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            if(i%5==0):
                print("likelyhood is ",self.log_likelihood())
            if(i%10==0):
                print("MPR is ",MPR(np.dot(self.user_vectors, self.item_vectors.T),self.R))
            if(i%2==0):
                user_vec_deriv, user_bias_deriv = self.deriv(True)
                user_vec_deriv_sum += user_vec_deriv**2
                user_bias_deriv_sum += user_bias_deriv**2
                vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
                bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)
                self.user_vectors += vec_step_size * user_vec_deriv
                self.user_biases += bias_step_size * user_bias_deriv
            else:
                # Fix users and solve for items
                # take step towards gradient of deriv of log likelihood
                # we take a step in positive direction because we are maximizing LL
                item_vec_deriv, item_bias_deriv = self.deriv(False)
                item_vec_deriv_sum += np.square(item_vec_deriv)
                item_bias_deriv_sum += np.square(item_bias_deriv)
                vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
                bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
                self.item_vectors += vec_step_size * item_vec_deriv
                self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()

            print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))

    def deriv(self, user):
        if user:
            vec_deriv = np.dot(self.counts, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=1), 1)

        else:
            vec_deriv = np.dot(self.counts.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=0), 1)

        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases.T
        A += self.item_biases
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.counts + self.ones) * A

        if user:
            vec_deriv -= np.dot(A, self.item_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.user_vectors
        else:
            vec_deriv -= np.dot(A.T, self.user_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases.T
        A += self.item_biases
        B = A * self.counts
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.counts + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()

        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()

def MPR(R_hat,R):
    rank = 0.0
    R_sum=np.sum(R)
    n_games=R.shape[1]
    R_hat_rank=np.argsort(np.argsort(-R_hat,axis=1))
    A=R*(R_hat_rank/n_games)
    rank = np.sum(A) / R_sum
    return rank

if __name__ == '__main__':
    train_data = pd.read_table('data1.txt', header=None)
    test_data = pd.read_table('test1.txt', header=None)
    n_users=len(set(train_data[0].tolist()+test_data[0].tolist()))
    n_items=len(set(train_data[1].tolist()+test_data[1].tolist()))
    print('There are {0} users and {1} games in the data'.format(n_users, n_items))

    user2idx = {user: i for i, user in enumerate( set(train_data[0].tolist()+test_data[0].tolist()) )}
    idx2user = {i: user for user, i in user2idx.items()}

    item2idx = {game: i for i, game in enumerate( set(train_data[1].tolist()+test_data[1].tolist()) )}
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

    LogMF = LogisticMF(C, R_test, 30)
    # print("start likelyhood is ",LogMF.log_likelihood())
    LogMF.train_model()
    print("end likelyhood is ", LogMF.log_likelihood())
