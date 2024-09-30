from conditional_feinsum import init_spn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
import numpy as np
from nns import MLP
import config
import argparse
import os

def eval_einsum(model_dir, model_id, dataset, device_id):
    if dataset == 'imagenet':
        transform = Compose([ToTensor(), Resize(112, antialias=True), CenterCrop(112)])
        ds = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split='val')
    elif dataset == 'imagenet32':
        transform = Compose([ToTensor(), Resize(32, antialias=True), CenterCrop(32)])
        ds = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split='val')
    elif dataset == 'celeba':
        transform = Compose([ToTensor(), Resize(64, antialias=True), CenterCrop(64)])
        ds = CelebA('/storage-01/datasets/', transform=transform, split='test')
    loader = DataLoader(ds, 16, num_workers=2)
    model_file = f'chk_{model_id}.pt'
    device = torch.device(f'cuda:{device_id}')
    einet = torch.load(model_dir + model_file).to(device)
    einet_lls = []
    for i, (x, y) in enumerate(loader):
        if i % 50 == 0:
            print(f"{(i / len(loader) * 100):3f}%")
        x = x.to(device)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
        ll_sample = einet.forward(x)
        einet_lls.append(ll_sample.detach().cpu().numpy().flatten())
    return np.concatenate(einet_lls)
    
    #samples = einet.sample(9).reshape(-1, config.height, config.width, 3)
    #save_image_stack(samples.cpu(), 3, 3, os.path.join(sample_dir, 'samples.png'))

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--device', type=int)
parser.add_argument('--cluster-file', default=None)

args = parser.parse_args()

if __name__ == '__main__':

    device = args.device

    lls = []

    if args.cluster_file is not None:
        clusters = np.load(args.cluster_file)

    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pt')]
    weights = []
    for i in range(len(model_files)):
        einet_lls = eval_einsum(args.model_dir, i, args.dataset, args.device)
        if args.cluster_file is not None:
            w = len(clusters[clusters == i]) / len(clusters)
            weighted_lls = einet_lls - np.log(w)
            lls.append(weighted_lls)
        else:
            lls.append(einet_lls)

    if args.cluster_file is not None:
        print(np.array(lls).shape)
        lls = torch.from_numpy(np.array(lls).T)
        lls = torch.logsumexp(lls, dim=1)
        print(lls.shape)
        print(lls.mean())
    else:
        print(lls[0].shape)
        print(lls[0].mean())
