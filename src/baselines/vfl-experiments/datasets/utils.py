import numpy as np
from .datasets import Income, BreastCancer, GimmeCredit
import torch

def get_vertical_data(ds, num_clients, rand_perm=False):
    
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            train_dataset = Income('../../../datasets/income/', split='train')
            test_dataset = Income('../../../datasets/income/', split='test')
        elif ds == 'breast-cancer':
            train_dataset = BreastCancer('../../../datasets/breast-cancer/', split='train')
            test_dataset = BreastCancer('../../../datasets/breast-cancer/', split='test')
        elif ds == 'credit':
            train_dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='train')
            test_dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='test')
        train_features = train_dataset.features
        train_targets = train_dataset.targets.reshape(-1, 1)
        test_features = test_dataset.features
        test_targets = test_dataset.targets.reshape(-1, 1)
        columns = train_features.shape[1]
        cols = np.arange(columns)
        if rand_perm:
            cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        split_cols = [list(s) for s in split_cols]
        client_train_data = [train_features[:, s] for s in split_cols]
        client_test_data = [test_features[:, s] for s in split_cols]
        return client_train_data, train_targets, client_test_data, test_targets, split_cols