import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip
import subprocess
import csv
import scipy.io as sp
from torch.utils.data import Subset, random_split
import torchvision
import json
import torch
import math


def maybe_download(directory, url_base, filename):
    filepath = os.path.join(directory, filename)
    print(filepath)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def maybe_download_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/mnist', 'http://yann.lecun.com/exdb/mnist/', file):
            continue
        print('unzip ../data/mnist/{}'.format(file))
        filepath = os.path.join('../data/mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_mnist():
    """Load MNIST"""

    maybe_download_mnist()

    data_dir = '../data/mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
            continue
        print('unzip ../data/fashion-mnist/{}'.format(file))
        filepath = os.path.join('../data/fashion-mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist():
    """Load fashion-MNIST"""

    maybe_download_fashion_mnist()

    data_dir = '../data/fashion-mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_debd():
    if os.path.isdir('../data/debd'):
        return
    subprocess.run(['git', 'clone', 'https://github.com/arranger1044/DEBD', '../data/debd'])
    wd = os.getcwd()
    os.chdir('../data/debd')
    subprocess.run(['git', 'checkout', '80a4906dcf3b3463370f904efa42c21e8295e85c'])
    subprocess.run(['rm', '-rf', '.git'])
    os.chdir(wd)


def load_debd(name, dtype='int32'):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    maybe_download_debd()

    data_dir = '../data/debd'

    train_path = os.path.join(data_dir, 'datasets', name, name + '.train.data')
    test_path = os.path.join(data_dir, 'datasets', name, name + '.test.data')
    valid_path = os.path.join(data_dir, 'datasets', name, name + '.valid.data')

    reader = csv.reader(open(train_path, 'r'), delimiter=',')
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, 'r'), delimiter=',')
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, 'r'), delimiter=',')
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


DEBD = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd',
        'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

DEBD_shapes = {
    'accidents': dict(train=(12758, 111), valid=(2551, 111), test=(1700, 111)),
    'ad': dict(train=(2461, 1556), valid=(491, 1556), test=(327, 1556)),
    'baudio': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'bbc': dict(train=(1670, 1058), valid=(330, 1058), test=(225, 1058)),
    'bnetflix': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'book': dict(train=(8700, 500), valid=(1739, 500), test=(1159, 500)),
    'c20ng': dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
    'cr52': dict(train=(6532, 889), valid=(1540, 889), test=(1028, 889)),
    'cwebkb': dict(train=(2803, 839), valid=(838, 839), test=(558, 839)),
    'dna': dict(train=(1600, 180), valid=(1186, 180), test=(400, 180)),
    'jester': dict(train=(9000, 100), valid=(4116, 100), test=(1000, 100)),
    'kdd': dict(train=(180092, 64), valid=(34955, 64), test=(19907, 64)),
    'kosarek': dict(train=(33375, 190), valid=(6675, 190), test=(4450, 190)),
    'moviereview': dict(train=(1600, 1001), valid=(250, 1001), test=(150, 1001)),
    'msnbc': dict(train=(291326, 17), valid=(58265, 17), test=(38843, 17)),
    'msweb': dict(train=(29441, 294), valid=(5000, 294), test=(3270, 294)),
    'nltcs': dict(train=(16181, 16), valid=(3236, 16), test=(2157, 16)),
    'plants': dict(train=(17412, 69), valid=(3482, 69), test=(2321, 69)),
    'pumsb_star': dict(train=(12262, 163), valid=(2452, 163), test=(1635, 163)),
    'tmovie': dict(train=(4524, 500), valid=(591, 500), test=(1002, 500)),
    'tretail': dict(train=(22041, 135), valid=(4408, 135), test=(2938, 135)),
    'voting': dict(train=(1214, 1359), valid=(350, 1359), test=(200, 1359)),
}

DEBD_display_name = {
    'accidents': 'accidents',
    'ad': 'ad',
    'baudio': 'audio',
    'bbc': 'bbc',
    'bnetflix': 'netflix',
    'book': 'book',
    'c20ng': '20ng',
    'cr52': 'reuters-52',
    'cwebkb': 'web-kb',
    'dna': 'dna',
    'jester': 'jester',
    'kdd': 'kdd-2k',
    'kosarek': 'kosarek',
    'moviereview': 'moviereview',
    'msnbc': 'msnbc',
    'msweb': 'msweb',
    'nltcs': 'nltcs',
    'plants': 'plants',
    'pumsb_star': 'pumsb-star',
    'tmovie': 'each-movie',
    'tretail': 'retail',
    'voting': 'voting'}


def maybe_download_svhn():
    svhn_files = ['train_32x32.mat', 'test_32x32.mat', "extra_32x32.mat"]
    for file in svhn_files:
        maybe_download('../../../datasets/svhn/', 'http://ufldl.stanford.edu/housenumbers/', file)


def load_svhn(dtype=np.uint8):
    """
    Load the SVHN dataset.
    """

    maybe_download_svhn()

    data_dir = '../../../datasets/svhn/'

    data_train = sp.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    data_test = sp.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    data_extra = sp.loadmat(os.path.join(data_dir, "extra_32x32.mat"))

    train_x = data_train["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    train_labels = data_train["y"].reshape(-1)

    test_x = data_test["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    test_labels = data_test["y"].reshape(-1)

    extra_x = data_extra["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    extra_labels = data_extra["y"].reshape(-1)

    return train_x, train_labels, test_x, test_labels, extra_x, extra_labels

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
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
        self.train_data = torchvision.datasets.ImageNet('../../../../datasets/imagenet/', split='train', transform=transform)
        self.val_data = torchvision.datasets.ImageNet('../../../../datasets/imagenet/', split='val', transform=transform)

class SVHNLoader(Loader):

    def __init__(self, n_clients, indspath, skew=0) -> None:
        super().__init__(n_clients, indspath, skew)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
        self.train_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='train', transform=transform, download=True)
        self.val_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='test', transform=transform, download=True)
        self.ex_data = torchvision.datasets.SVHN('../../../../datasets/svhn/', split='extra', transform=transform, download=True)

class CelebaLoader(Loader):

    def __init__(self, n_clients, indspath, skew=0) -> None:
        super().__init__(n_clients, indspath, skew)
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
        self.train_data = torchvision.datasets.CelebA('../../../../datasets/celeba/', split='train', transform=transform)
        self.val_data = torchvision.datasets.CelebA('../../../../datasets/celeba/', split='valid', transform=transform)


def get_dataset_loader(dataset, num_clients, indspath, skew=0):
    if dataset == 'mnist':
        return FashionMNISTLoader(num_clients, indspath, skew=skew)
    elif dataset == 'imagenet':
        return ImageNetLoader(num_clients, indspath, skew)
    elif dataset == 'svhn':
        return SVHNLoader(num_clients, indspath, skew=skew)
    elif dataset == 'celeba':
        return CelebaLoader(num_clients, indspath, skew=skew)
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
