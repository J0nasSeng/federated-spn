from .datasets import TabularDataset
import numpy as np
from datasets.datasets import Avazu, Income
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch
from fedlab.utils.dataset import BasicPartitioner, MNISTPartitioner

def get_horizontal_train_data(ds, num_clients, partitioning='iid', dir_alpha=0.2):
    if ds == 'income':
        dataset = Income('../../datasets/income/', split='train')
        partitioner = IncomePartitioner(dataset.targets, num_clients, 
                                        partition=partitioning, dir_alpha=dir_alpha)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
    elif ds == 'mnist':
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = MNIST('../../datasets/avazu/', True, transform=transform, download=True)
        partitioner = MNISTPartitioner(dataset.targets, num_clients, 
                                       partition=partitioning, dir_alpha=dir_alpha)
        imgs = dataset.data.reshape((-1, 28*28)).numpy().astype(np.float64)
        imgs /= 255.
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets)).astype(np.float64)
    elif ds == 'avazu':
        dataset = Avazu('../../datasets/avazu/', split='train')
        partitioner = AvazuPartitioner(dataset.targets, num_clients, 
                                       partition=partitioning, dir_alpha=dir_alpha)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))

    partitioned_data = []
    for _, idx in partitioner.client_dict.items():
        partitioned_data.append(data[idx])
    return partitioned_data

def get_vertical_train_data(ds, num_clients):
    if ds == 'income':
        columns = 15
        if num_clients > 5:
            raise ValueError("'num_clients' must be smaller than 6.")
        cols = np.arange(columns)
        cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        split_cols = [list(s) for s in split_cols]

        dataset = Income('../../datasets/income/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        client_data = [data[:, s] for s in split_cols]
        return client_data, split_cols

    elif ds == 'mnist':
        columns = (28*28) + 1
        if np.floor(columns / num_clients) < 3:
            raise ValueError("'num_clients' too high")
        cols = np.arange(columns)
        cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = MNIST('../../datasets/', True, transform=transform, download=True)
        imgs = dataset.data.reshape((-1, 28*28)).numpy().astype(np.float64)
        imgs /= 255.
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets)).astype(np.float64)
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
    min_dims, max_dims = np.ceil(min_dim_frac*num_cols), np.ceil(max_dim_frac*num_cols)
    cols = np.arange(num_cols)
    client_cols = []
    for c in range(num_clients):
        subspace = np.random.choice(cols, np.random.randint(min_dims, max_dims))
        client_cols.append(set(subspace))
    union = set().union(*client_cols)
    if not len(union) == num_cols:
        missing = [c for c in cols if c not in union]
        for m in missing:
            client = np.random.randint(0, num_clients)
            client_cols[client] = list(client_cols[client] + [m])
    client_data = []
    idx = np.arange(data.shape[0])
    idx = np.random.permutation(idx)
    client_idx = np.array_split(idx, num_clients)
    for c in range(num_clients):
        subspace = client_cols[c]
        cidx = client_idx[c]
        client_data.append(data[cidx, subspace])
    return client_data, client_cols
    
def get_hybrid_train_data(ds, num_clients, min_dim_frac=0.25, max_dim_frac=0.5,
                          sample_frac=None, seed=111):
    if ds == 'income':
        columns = 15
        if num_clients > 5:
            raise ValueError("'num_clients' must be smaller than 6.")

        dataset = Income('../../datasets/income/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        client_data, subspaces = split_dataset_hybrid(data, num_clients, columns, 
                                                      min_dim_frac, max_dim_frac, sample_frac, seed)
        return client_data, subspaces

    elif ds == 'mnist':
        columns = (28*28) + 1
        if np.floor(columns / num_clients) < 3:
            raise ValueError("'num_clients' too high")
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = MNIST('../../datasets/', True, transform=transform, download=True)
        imgs = dataset.data.reshape((-1, 28*28)).numpy().astype(np.float64)
        imgs /= 255.
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets)).astype(np.float64)
        client_data, subspaces = split_dataset_hybrid(data, num_clients, columns, 
                                                      min_dim_frac, max_dim_frac, sample_frac, seed)
        return client_data, subspaces
    
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
    if ds == 'income':
        dataset = Income('../../datasets/income/', split='test')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        return data
    elif ds == 'mnist':
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = MNIST('../../datasets/', False, transform=transform, download=True)
        imgs = dataset.data.reshape((-1, 28*28)).numpy()
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets))
        return data
    elif ds == 'avazu':
        dataset = Avazu('../../datasets/', split='test')
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
        x, y = d[...,:-1], d[...,-1]
        tds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        tdl = DataLoader(tds, batch_size=batch_size)
        return tdl
        
class IncomePartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 14

class AvazuPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 16