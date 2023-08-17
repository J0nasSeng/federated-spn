from .datasets import TabularDataset
import numpy as np
from datasets.datasets import Avazu, Income
from torchvision.datasets import MNIST
import torchvision
from fedlab.utils.dataset import BasicPartitioner, MNISTPartitioner

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

def get_train_data(ds, num_clients, partitioning='iid'):
    if ds == 'income':
        dataset = Income('../../datasets/income/', split='train')
        partitioner = IncomePartitioner(dataset.targets, num_clients, partition=partitioning)
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))
    elif ds == 'mnist':
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = MNIST('../../datasets/', True, transform=transform, download=True)
        partitioner = MNISTPartitioner(dataset.targets, num_clients, partition=partitioning)
        imgs = dataset.data.reshape((-1, 28*28)).numpy().astype(np.float64)
        imgs /= 255.
        targets = dataset.targets.reshape((-1, 1)).numpy()
        data = np.hstack((imgs, targets)).astype(np.float64)
    elif ds == 'avazu':
        dataset = Avazu('../../datasets/', split='train')
        np_features = dataset.features.numpy()
        np_targets = dataset.targets.numpy()
        data = np.hstack((np_features, np_targets.reshape(-1, 1)))

    partitioned_data = []
    for _, idx in partitioner.client_dict.items():
        partitioned_data.append(data[idx])
    return partitioned_data
    
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
        
class IncomePartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 14

class AvazuPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 16