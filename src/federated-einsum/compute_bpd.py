import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
import config
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bits_per_dim_cont(ll, x, lambda_=0.05, dims=config.num_vars * config.num_dims):
    """
    neg_ll: negative LL, shape=(B,)
    x: batch of images, shape=(B,*)
    """
    frac = -ll / (dims * np.log(2))
    num_bin_const = 8  # assumes 256 bins because we have RGB images with 256 possible values per dimension
    log_sigmoid_transformed_x = np.log2(sigmoid(x)) + np.log2(1 - sigmoid(x))
    bpd = (
        frac
        - np.log2(1 - 2 * lambda_)
        + num_bin_const
        + ((1 / dims) * np.sum(log_sigmoid_transformed_x, axis=1))
    )
    return bpd


def bits_per_dim(nll, num_features):
    # If NLL is summed across a batch, take the mean
    nll_mean = nll.mean() if isinstance(nll, torch.Tensor) else np.mean(nll)

    # Compute bits per dimension
    bpd = nll_mean / (num_features * np.log(2))
    return bpd


dataset = "celeba"
device = torch.device("cuda:0")
# lls = torch.load('./lls_im32').to(device)

if dataset == "imagenet":
    transform = Compose([ToTensor(), Resize(112, antialias=True), CenterCrop(112)])
    ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform, split="val")
elif dataset == "imagenet32":
    transform = Compose([ToTensor(), Resize(32, antialias=True), CenterCrop(32)])
    ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform, split="val")
elif dataset == "celeba":
    transform = Compose([ToTensor(), Resize(64, antialias=True), CenterCrop(64)])
    ds = CelebA("/storage-01/datasets/", transform=transform, split="test")
loader = DataLoader(ds, 32, num_workers=2, shuffle=False)

bits_per_dimension = []
for i, (x, y) in enumerate(loader):
    if i % 50 == 0:
        print(f"{(i / len(loader) * 100):3f}%")
    x = x.to(device)
    x = x.permute((0, 2, 3, 1))
    x = x.reshape(x.shape[0], config.num_vars * config.num_dims)
    start_idx = i * loader.batch_size
    end_idx = (i + 1) * loader.batch_size
    # lls_batch = lls[start_idx:end_idx].detach().cpu().numpy()
    lls_batch = np.repeat([664], x.shape[0]).flatten()
    bpd = bits_per_dim_cont(lls_batch, x.detach().cpu().numpy(), lambda_=0)
    bits_per_dimension.append(bpd)

bits_per_dimension = np.concatenate(bits_per_dimension)
print(np.mean(bits_per_dimension))
