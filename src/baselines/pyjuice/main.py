import torch
import pyjuice as juice
import pyjuice.nodes.distributions as juice_dists
from pyjuice.structures import PD, RAT_SPN
from torchvision.datasets import ImageNet, SVHN, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from rtpt import RTPT
import numpy as np

batch_size = 64

def load_dataset(ds_name, split='train'):
    if ds_name == 'imagenet':
        transform = Compose([ToTensor(), Resize(112, antialias=True), CenterCrop(112)])
        dataset = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split=split)
        data_shape = (112, 112, 3)
    elif ds_name == 'imagenet32':
        transform = Compose([ToTensor(), Resize(32, antialias=True), CenterCrop(32)])
        dataset = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split=split)
        data_shape = (32, 32, 3)
    elif ds_name == 'celeba':
        transform = Compose([ToTensor(), Resize(32, antialias=True), CenterCrop(32)])
        dataset = CelebA('/storage-01/datasets/', transform=transform, split=split)
        data_shape = (32, 32, 3)
    return dataset, data_shape

def train(ds_name, num_epochs):

    rt = RTPT('JS', 'FedEinsum', num_epochs)
    rt.start()
    device = torch.device(f'cuda:{0}')
    dataset, shape = load_dataset(ds_name)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    arch = RAT_SPN(np.prod(list(shape)), 256, 2, 6, input_node_type=juice_dists.Gaussian, input_node_params={'mu': 0.0, 'sigma': 1.0, 'min_sigma': 1e-6})
    #arch = PD(shape, 256, input_dist=juice_dists.Gaussian(0.0, 0.1, 1e-6), split_points=[[64//4], [64//4], [1]])
    print(arch)
    model = juice.compile(arch)
    model = model.to(device)

    for e in range(num_epochs):

        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            x = x.reshape(x.shape[0], -1)

            # This is equivalent to zeroing out the parameter gradients of a neural network
            model.init_param_flows(flows_memory = 0.0)
            # Forward pass
            lls = model(x)
            # Backward pass
            lls.mean().backward()
            # Mini-batch EM
            model.mini_batch_em(step_size = 0.01, pseudocount = 0.001)

            if i % 50 == 0:
                print(f"Epoch {e}/{num_epochs}: \t Iter: {i}/{len(loader)}: \t LL: {lls.mean()}")
    
    return model


def evaluate(model, dataset):

    loader = DataLoader(dataset, batch_size=256, num_workers=2)

    avg_lls = []

    for x, y in loader:

        lls = model(x)
        avg_lls.append(lls.detach().cpu().mean().numpy())

    return np.sum(avg_lls) / len(loader)


model = train('celeba', 10)
test_set = load_dataset('celeba', 'test')
result = evaluate(model, test_set)
print(result)
