from .datasets import TabularDataset
import numpy as np
from datasets.datasets import DatasetFactory
from datasets.partitioners import PartitionerFactory
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch
from fedlab.utils.dataset import MNISTPartitioner

def get_horizontal_train_data(ds, num_clients, partitioning='iid', dir_alpha=0.2):
    if ds in ['income', 'breast-cancer', 'credit', 'baf']:
        dataset_factory = DatasetFactory()
        dataset = dataset_factory.load_dataset(ds)
        dataset.set_split('train')
        partitioner_factory = PartitionerFactory()
        Partitioner_cls = partitioner_factory.get_partitioner_cls(dataset)
        partitioner = Partitioner_cls(dataset.targets, num_clients, partition=partitioning, dir_alpha=dir_alpha)
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

    partitioned_data = []
    for _, idx in partitioner.client_dict.items():
        partitioned_data.append(data[idx])
    return partitioned_data

def get_vertical_train_data(ds, num_clients, rand_perm=True, return_labels=False):
    
    if ds in ['income', 'breast-cancer', 'credit']:
        dataset_factory = DatasetFactory()
        dataset = dataset_factory.load_dataset(ds)
        dataset.set_split('train')

        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        if not return_labels:
            data = np.hstack([features, targets.reshape(-1, 1)])
        else:
            data = features
        columns = data.shape[1]
        cols = np.arange(columns)
        if rand_perm:
            cols = np.random.permutation(cols)
        split_cols = np.array_split(cols, num_clients)
        split_cols = [list(s) for s in split_cols]
        client_data = [data[:, s] for s in split_cols]
        if not return_labels:
            return client_data, split_cols
        else:
            return client_data, split_cols, targets

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
    
def split_dataset_hybrid(data, num_clients, num_cols, overlap_frac, sample_frac, seed):
    sample_frac = 1/num_clients if sample_frac is None else sample_frac
    #np.random.seed(seed)
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
    client_overlap = np.random.choice(idx, int(overlap_frac*len(idx)))
    client_indices = []
    for c in range(num_clients):
        subspace = np.array(client_cols[c])
        subsample_size = int((1 / num_clients) * len(data))
        client_idx = np.random.choice(idx, subsample_size)
        client_idx = np.concatenate((client_idx, client_overlap))
        client_idx_client_view = np.arange(len(client_idx))
        subspace_data = data[:, subspace]
        c_data = subspace_data[client_idx]
        client_data.append(c_data)
        client_indices.append((client_idx, client_idx_client_view))
    return client_data, client_cols, client_indices
    
def get_hybrid_train_data(ds, num_clients, overlap_frac=0.3,
                          sample_frac=None, seed=111, return_labels=False):
    if ds in ['income', 'breast-cancer', 'credit']:
        dataset_factory = DatasetFactory()
        dataset = dataset_factory.load_dataset(ds)
        dataset.set_split('train')

        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        if not return_labels:
            data = np.hstack([features, targets.reshape(-1, 1)])
        else:
            data = features
        client_data, subspaces, client_idx = split_dataset_hybrid(data, num_clients, data.shape[1], 
                                                      overlap_frac, sample_frac, seed)
        if not return_labels:
            return client_data, subspaces, client_idx
        else:
            return client_data, subspaces, client_idx, targets

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
        client_data, subspaces, client_idx = split_dataset_hybrid(data, num_clients, columns, 
                                                      overlap_frac, sample_frac, seed)
        return client_data, subspaces, client_idx
    
def get_test_data(ds):
    if ds in ['income', 'breast-cancer', 'credit', 'baf']:
        dataset_factory = DatasetFactory()
        dataset = dataset_factory.load_dataset(ds)
        dataset.set_split('test')

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
    