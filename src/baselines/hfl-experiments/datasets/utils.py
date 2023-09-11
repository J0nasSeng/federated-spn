from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
from fedlab.utils.dataset import BasicPartitioner, MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.dataset.partition import BasicPartitioner
from .datasets import Avazu, Income, GimmeCredit, BreastCancer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torchvision

class IncomePartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 14

class AvazuPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 16

class CreditPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 10

class BreastCancerPartitioner(BasicPartitioner):

    num_classes = 2
    num_features = 30

class PartitionedIncome:

    def __init__(self, num_clients, partitioning, root='../../../datasets/income/', split='train'):
        self.dataset = Income(root, split=split)
        self.partitioner = IncomePartitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 14
        self.out_dim = 2

    def get_dataset(self, client_num):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds
    
    def get_dataloader(self, id, batch_size):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx], targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader
    
class PartitionedAvazu:

    def __init__(self, num_clients, partitioning, root='../../../datasets/avazu/', split='train'):
        self.dataset = Avazu(root, split=split)
        self.partitioner = AvazuPartitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 20
        self.out_dim = 2

    def get_dataset(self, client_num):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds

    def get_dataloader(self, id, batch_size):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx], targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader
    
class PartitionedBreastCancer:
    def __init__(self, num_clients, partitioning, root='../../../datasets/breast-cancer/', split='train'):
        self.dataset = BreastCancer(root, split=split)
        self.partitioner = BreastCancerPartitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 30
        self.out_dim = 2

    def get_dataset(self, client_num):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds

    def get_dataloader(self, id, batch_size):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx], targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader

class PartitionedCredit:

    def __init__(self, num_clients, partitioning, root='../../../datasets/GiveMeSomeCredit/', split='train'):
        self.dataset = GimmeCredit(root, split=split)
        self.partitioner = CreditPartitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 10
        self.out_dim = 2

    def get_dataset(self, client_num):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds

    def get_dataloader(self, id, batch_size):
        features = self.dataset.features.to(torch.float32)
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx], targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader
    
class PartitionedMNIST:

    def __init__(self, num_clients, partitioning, root='../../../datasets/', train=True):
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        self.dataset = MNIST(root, train, transform=transform, download=True)
        self.partitioner = MNISTPartitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 28*28
        self.out_dim = 10

    def get_dataset(self, client_num):
        features = self.dataset.data / 255.
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds
    
    def get_dataloader(self, id, batch_size):
        features = self.dataset.data / 255.
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx].unsqueeze(1), targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader
    
class PartitionedCIFAR10:

    def __init__(self, num_clients, partitioning, root='../../../datasets/cifar10', train=True):
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        self.dataset = CIFAR10(root, train, transform=transform, download=True)
        self.partitioner = CIFAR10Partitioner(self.dataset.targets, num_clients, partition=partitioning)
        self.in_dim = 32*32
        self.out_dim = 10

    def get_dataset(self, client_num):
        features = self.dataset.data / 255.
        targets = self.dataset.targets
        idx = self.partitioner.client_dict[client_num]
        ds = TensorDataset(features[idx], targets[idx])
        return ds
    
    def get_dataloader(self, id, batch_size):
        features = torch.from_numpy(self.dataset.data) / 255.
        features = features.permute(0, 3, 1, 2)
        targets = torch.tensor(self.dataset.targets)
        idx = self.partitioner.client_dict[id]
        ds = TensorDataset(features[idx], targets[idx])
        loader = DataLoader(ds, batch_size)
        return loader

def get_horizontal_train_data(ds, num_clients, partitioning='iid'):
    if ds == 'income':
        dataset = PartitionedIncome(num_clients, partitioning=partitioning)
        return dataset
    elif ds == 'breast-cancer':
        dataset = PartitionedBreastCancer(num_clients, partitioning)
        return dataset
    elif ds == 'credit':
        dataset = PartitionedCredit(num_clients, partitioning)
        return dataset
    elif ds == 'mnist':
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        dataset = PartitionedMNIST(root='../../../datasets/',
                                   num_clients=num_clients,
                                   partitioning=partitioning)
        return dataset
    elif ds == 'avazu':
        return PartitionedAvazu(num_clients, partitioning)
    elif ds == 'cifar10':
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        return PartitionedCIFAR10(root='../../../datasets/cifar10/',
                                  num_clients=num_clients,
                                  partitioning=partitioning)
    
def get_test_dataset(ds):
    if ds == 'income':
        income = Income('../../../datasets/income/', 'test')
        return income.dataset
    elif ds == 'avazu':
        avazu = Avazu('../../../datasets/avazu/', 'test')
        return avazu.dataset
    elif ds == 'breast-cancer':
        return BreastCancer('../../../datasets/breast-cancer/', 'test').dataset
    elif ds == 'credit':
        return GimmeCredit('../../../datasets/GiveMeSomeCredit/', 'test').dataset
    elif ds == 'mnist':
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        mnist = torchvision.datasets.MNIST(root="../../../datasets/",
                                        train=False,
                                        download=True,
                                        transform=transform)
        return mnist
    elif ds == 'cifar10':
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                             ])
        cifar = torchvision.datasets.CIFAR10(root="../../../datasets/cifar10",
                                        train=False,
                                        transform=transform)
                                        
        return cifar
    