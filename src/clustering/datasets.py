import numpy as np
from torch.utils.data import Subset, random_split
import torchvision
import json
import torch
import math
import os


class Loader:

    def __init__(self, n_clients, indspath, skew=0) -> None:
        self.n_clients = n_clients
        self.skew = skew
        self.indspath = indspath
        self.train_data = None
        self.val_data = None

    def partition(self, reuse_json=True):
        """
        Loads the Fashion-MNIST dataset
        """
        if reuse_json and os.path.isfile(self.indspath):
            print("Reusing existing indices")
            return
        self.train_partitions, self.val_partitions, self.test_set, train_inds, val_inds, test_inds = partition_skewed(self.train_data, self.val_data, self.n_clients, skew=self.skew)
        train_dict, val_dict = {}, {}
        for i, inds in enumerate(train_inds):
            train_dict[i] = inds.tolist()
        for i, inds in enumerate(val_inds):
            val_dict[i] = inds.tolist()
        json_dict = {
            'train': train_dict,
            'val': val_dict,
            'test': test_inds.tolist()
        }
        with open(self.indspath, 'w+') as f:
            json.dump(json_dict, f)

    def load_client_data(self, client_id):
        with open(self.indspath, 'r') as f:
            inds_dict = json.load(f)
        train_inds = np.array(inds_dict['train'][str(client_id)])
        val_inds = np.array(inds_dict['val'][str(client_id)])
        trainset = Subset(self.train_data, train_inds)
        valset = Subset(self.val_data, val_inds)
        return trainset, valset

    def load_server_data(self):
        with open(self.indspath, 'r') as f:
            inds_dict = json.load(f)
        test_inds = np.array(inds_dict['test'])
        testset = Subset(self.val_data, test_inds)
        return testset
         
    def get_client_data(self):
        for train, val in zip(self.train_partitions, self.val_partitions):
            yield train, val

    def get_test(self):
        return self.test_set

class FashionMNISTLoader(Loader):

    def __init__(self, n_clients, indspath, skew=0) -> None:
        super().__init__(n_clients, indspath, skew)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
        self.train_data = torchvision.datasets.FashionMNIST('../../../../datasets/femnist/', download=True, train=True, transform=transform)
        self.val_data = torchvision.datasets.FashionMNIST('../../../../datasets/femnist/', download=True, train=False, transform=transform)

class ImageNetLoader(Loader):

    def __init__(self, n_clients, indspath, skew=0) -> None:
        super().__init__(n_clients, indspath, skew)
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
        self.train_data = torchvision.datasets.ImageNet('../../../../datasets/imagenet/', split='train', transform=transform)
        self.val_data = torchvision.datasets.ImageNet('../../../../datasets/imagenet/', split='val', transform=transform)

class SVHNLoader(Loader):

    def __init__(self, n_clients, indspath, skew=0) -> None:
        super().__init__(n_clients, indspath, skew)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
        self.train_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='train', transform=transform, download=True)
        self.val_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='test', transform=transform, download=True)
        self.ex_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='extra', transform=transform, download=True)


def get_dataset_loader(dataset, num_clients, indspath, skew=0):
    if dataset == 'mnist':
        return FashionMNISTLoader(num_clients, indspath, skew=skew)
    elif dataset == 'imagenet':
        return ImageNetLoader(num_clients, indspath, skew)
    elif dataset == 'svhn':
        return SVHNLoader(num_clients, indspath, skew=skew)
    else:
        raise ValueError('{} is not supported'.format(dataset))

def partition_data(train_set, val_set, n_clients):
    train_len = len(train_set)
    val_len = len(val_set) // 2
    
    # split validation set into validation and test set
    val_inds = np.arange(val_len)
    test_inds = np.array(range(val_len, 2*val_len))
    validation = Subset(val_set, val_inds)
    test = Subset(val_set, test_inds)

    # split sets in n_clients random and non-overlapping samples
    train_lengths = np.repeat(train_len // n_clients, n_clients)
    val_lengths = np.repeat(val_len // n_clients, n_clients)
    train_partitions = random_split(train_set, train_lengths, generator=torch.Generator().manual_seed(42))
    val_partitions = random_split(validation, val_lengths, generator=torch.Generator().manual_seed(42))

    return train_partitions, val_partitions, test

def label_distribution_skew(y, partitions, skew=1):
    def runner_split(N_labels, N_runners):
        """number of labels to assign to n clients"""
        runner_labels = round(max(1, N_labels / N_runners))
        runner_split = round(max(1, N_runners / N_labels))
        return runner_labels, runner_split

    runn_inds = []
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    N_labels = torch.unique(y).shape[0]
    n_labels, n_runners = runner_split(N_labels, partitions)
    np.random.seed(42)
    
    selected_inds = []
    for label_idx in range(0, N_labels, n_labels):
        # get a range of labels (e.g. in case of MNIST labels between 0 and 3)
        mask = torch.isin(y, torch.Tensor(np.arange(label_idx, label_idx+n_labels))).cpu().detach().numpy()
        # get index of these labels
        subset_idx = np.argwhere(mask)[:, 0]
        n_samples = subset_idx.shape[0]
        # randomly sample indices of size sample_size
        sample_size = math.floor(skew*n_samples)
        subset_idx = np.random.choice(subset_idx, sample_size, replace=False)
        selected_inds += list(subset_idx)
    
        for partition in np.array_split(subset_idx, n_runners):
            # add a data-partition for a runner
            runn_inds.append(partition)
    
    selected = np.array(selected_inds)
    mask = np.zeros(len(y))
    mask[selected] = 1
    not_selected = np.argwhere(mask == 0)
    return runn_inds, not_selected

def uniform_distribution(inds, partitions, randomise=True):
    runner_data = []
    if randomise:
        # shuffle indices
        np.random.shuffle(inds)
    # split randomly chosen data-points and labels into partitions
    for partition in np.array_split(inds, partitions):
        runner_data.append(partition)
    return runner_data

def partition_skewed(train_set, val_set, partitions, randomise=True, skew=0):
    # randomly select half of the data of the validation set to be the test-set
    ind_in_val = np.random.choice([0, 1], p=[0.5, 0.5], size=len(val_set))
    val_inds = np.argwhere(ind_in_val == 1).reshape(1, -1)[0]
    test_inds = np.argwhere(ind_in_val == 0).reshape(1, -1)[0]
    train_partitions, val_partitions, test_set = [], [], Subset(val_set, test_inds)
    train_inds = np.arange(len(train_set))
    train_subs_inds, val_subs_inds, test_subs_inds = [], [], test_inds
    if skew == 0:
        train_unfiorm = uniform_distribution(train_inds, partitions, randomise)
        val_uniform = uniform_distribution(val_inds, partitions, randomise)
        for t, v in zip(train_unfiorm, val_uniform):
            train_subset = Subset(train_set, t)
            val_subset = Subset(val_set, v)
            train_partitions.append(train_subset)
            val_partitions.append(val_subset)
            train_subs_inds.append(t)
            val_subs_inds.append(v)
    else:
        # build skewed data-sets
        train_selected, train_inds_remain = label_distribution_skew(train_set.targets, partitions, skew)
        val_selected, val_inds_remain = label_distribution_skew(val_set.targets, partitions, skew)
        # if skew < 1 this will contain a list of all data-points that are not already assigned to some runner
        train_uniform = uniform_distribution(train_inds_remain, partitions, randomise)
        val_uniform = uniform_distribution(val_inds_remain, partitions, randomise)
        # concatenate train and val set-indices obtained above
        for s, p in zip(train_selected, train_uniform):
            s = s.reshape(1, -1)[0]
            p = p.reshape(1, -1)[0]
            indices = np.concatenate((s, p))
            subset = Subset(train_set, indices)
            train_partitions.append(subset)
            train_subs_inds.append(indices)
        for s, p in zip(val_selected, val_uniform):
            s = s.reshape(1, -1)[0]
            p = p.reshape(1, -1)[0]
            indices = np.concatenate((s, p))
            subset = Subset(val_set, indices)
            val_partitions.append(subset)
            val_subs_inds.append(indices)
    return train_partitions, val_partitions, test_set, train_subs_inds, val_subs_inds, test_subs_inds


if __name__ == '__main__':
    print('Downloading dataset -- this might take a while')

    print()
    print('MNIST')
    maybe_download_mnist()

    print()
    print('fashion MNIST')
    maybe_download_fashion_mnist()

    print()
    print('20 binary datasets')
    maybe_download_debd()

    print()
    print('SVHN')
    maybe_download_svhn()
