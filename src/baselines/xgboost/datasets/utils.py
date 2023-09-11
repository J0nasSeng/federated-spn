from .datasets import TabularDataset
import numpy as np
from datasets.datasets import Avazu, Income, BreastCancer, GimmeCredit
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch
from fedlab.utils.dataset import BasicPartitioner, MNISTPartitioner

def get_horizontal_train_data(ds, num_clients, partitioning='iid', dir_alpha=0.2):
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            dataset = Income('../../../datasets/income/', split='train')
            partitioner = IncomePartitioner(dataset.targets, num_clients, 
                                        partition=partitioning, dir_alpha=dir_alpha)
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../../datasets/breast-cancer/', split='train')
            partitioner = BreasCancerPartitioner(dataset.targets, num_clients,
                                                 partition=partitioning, dir_alpha=dir_alpha)
        elif ds == 'credit':
            dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='train')
            partitioner = GimmeCreditPartitioner(dataset.targets, num_clients,
                                                 partition=partitioning, dir_alpha=dir_alpha)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
    elif ds == 'avazu':
        dataset = Avazu('../../../datasets/avazu/', split='train')
        partitioner = AvazuPartitioner(dataset.targets, num_clients, 
                                       partition=partitioning, dir_alpha=dir_alpha)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))

    partitioned_data = []
    for _, idx in partitioner.client_dict.items():
        partitioned_data.append(data[idx])
    return partitioned_data

def get_train_data(ds):
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            dataset = Income('../../../datasets/income/', split='train')
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../../datasets/breast-cancer/', split='train')
        elif ds == 'credit':
            dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
    elif ds == 'avazu':
        dataset = Avazu('../../../datasets/avazu/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
    return data

def get_vertical_train_data(ds, num_clients, rand_perm=False):
    
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            dataset = Income('../../../datasets/income/', split='train')
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../../datasets/breast-cancer/', split='train')
        elif ds == 'credit':
            dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='train')
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        data = np.hstack([features, targets.reshape(-1, 1)])
        columns = data.shape[1]
        cols = np.arange(columns)
        if rand_perm:
            cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        split_cols = [list(s) for s in split_cols]
        client_data = [data[:, s] for s in split_cols]
        return client_data, split_cols
    
    elif ds == 'avazu':
        columns = 21
        if num_clients > 7:
            raise ValueError("'num_clients' must be smaller than 8.")
        cols = np.arange(columns)
        cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        dataset = Avazu('../../datasets/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        client_data = [data[:, s] for s in split_cols]
        return client_data, split_cols
    
def split_dataset_hybrid(data, num_clients, num_cols, min_dim_frac, max_dim_frac, sample_frac, seed):
    sample_frac = 1/num_clients if sample_frac is None else sample_frac
    np.random.seed(seed)
    cols_per_client = int(num_cols / num_clients)
    client_to_col = []
    for client_id in range(num_clients):
        client_to_col += [client_id] * cols_per_client
    
    if len(client_to_col) < num_cols:
        num_missing = num_cols - len(client_to_col)
        # randomly sample clients to add
        client_ids = np.random.choice(list(range(num_clients)), num_missing)
        client_to_col += list(client_ids)
    
    rand_assignment = np.random.permutation(client_to_col)
    client_col_assignment = {}
    for cid in range(num_clients):
        col_idx = np.argwhere(rand_assignment == cid).flatten()
        client_col_assignment[cid] = col_idx
    client_cols = [list(c) for c in client_col_assignment.values()]
    client_data = []
    idx = np.arange(data.shape[0])
    idx = np.random.permutation(idx)
    client_idx = np.array_split(idx, num_clients)
    for c in range(num_clients):
        subspace = np.array(client_cols[c])
        cidx = client_idx[c]
        subspace_data = data[:, subspace]
        c_data = subspace_data[cidx]
        client_data.append(c_data)
    return client_data, client_cols, client_idx
    
def get_hybrid_train_data(ds, num_clients, min_dim_frac=0.25, max_dim_frac=0.5,
                          sample_frac=None, seed=111):
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            dataset = Income('../../../datasets/income/', split='train')
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../../datasets/breast-cancer/', split='train')
        elif ds == 'credit':
            dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='train')
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        data = np.hstack([features, targets.reshape(-1, 1)])
        client_data, subspaces, client_idx = split_dataset_hybrid(data, num_clients, columns, 
                                                      min_dim_frac, max_dim_frac, sample_frac, seed)
        return client_data, subspaces, client_idx

    elif ds == 'avazu':
        columns = 21
        if num_clients > 7:
            raise ValueError("'num_clients' must be smaller than 8.")
        dataset = Avazu('../../datasets/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        client_data, subspaces = split_dataset_hybrid(data, num_clients, columns, 
                                                      min_dim_frac, max_dim_frac, sample_frac, seed)
        return client_data, subspaces
    
def get_test_data(ds):
    if ds in ['income', 'breast-cancer', 'credit']:
        if ds == 'income':
            dataset = Income('../../../datasets/income/', split='test')
        elif ds == 'breast-cancer':
            dataset = BreastCancer('../../../datasets/breast-cancer/', split='test')
        elif ds == 'credit':
            dataset = GimmeCredit('../../../datasets/GiveMeSomeCredit/', split='test')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        return data
    elif ds == 'avazu':
        dataset = Avazu('../../../datasets/', split='test')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        return data
    
def make_data_loader(ds, batch_size=64):
    if type(ds) is list:
        data_loaders = []
        for d in ds:
            x, y = d[...,:-1], d[...,-1]
            tds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
            tdl = DataLoader(tds, batch_size=batch_size)
            data_loaders.append(tdl)
        return data_loaders
    else:
        x, y = ds[...,:-1], ds[...,-1]
        tds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        tdl = DataLoader(tds, batch_size=batch_size)
        return tdl
        
class IncomePartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 14

class AvazuPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 16

class BreasCancerPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 31

class GimmeCreditPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 11