from .datasets import TabularDataset
import numpy as np
from datasets.datasets import Avazu, Income
from torchvision.datasets import MNIST
import torchvision

def split_tabular_vertical(dataset: TabularDataset, 
                   num_clients, p=None, seed=111):
    
    """
        Split a given tabular dataset vertically,
        i.e. distribute features across clients. 

        :param dataset: Dataset to be split
        :param num_clients: How many clients should dataset be split over?
        :param p: weighting of clients, how many featues should each client hold?
        :param seed: Seed for random feature choice
    """
    np.random.seed(seed)
    num_features = dataset.features.shape[1]
    assert num_clients <= num_features, 'Too many clients'

    if p is None:
        p = [1/num_clients for _ in range(num_clients)]

    assert len(p) == num_clients, 'p and num_clients must match'

    feature_idx = list(range(num_features))
    client_datasets = []
    for frac in p:
        s = int(np.ceil(frac * num_clients))
        idx = np.random.choice(feature_idx, size=s, replace=False)
        feature_idx = [f for f in feature_idx if f not in idx]

        x = dataset.features[:, idx]
        ds = TabularDataset(x, dataset.targets)
        client_datasets.append(ds)
    
    return client_datasets

def get_data(ds, split='train'):
    if ds == 'income':
        dataset = Income('../../datasets/income/', split=split)
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
        dataset = MNIST('../../datasets/mnist/', split == 'train', transform=transform)
        imgs = dataset.data.reshape((-1, 28*28)).numpy()
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets))
        return data
    elif ds == 'avazu':
        dataset = Avazu('../../datasets/income/', split=split)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
        return data