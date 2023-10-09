from nns import MLP, CNN
import numpy as np
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from ceinsum import EinsumNetwork, Graph, EinetMixture
import config
import torch
import os
import logging
import sys
from utils import save_image_stack
from sklearn.cluster import KMeans
from multiprocessing import Process
import pickle
from rtpt import RTPT
from utils import extract_image_patches, get_surrounding_patches, set_einet_weights
import torch.nn.functional as F

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

def init_spn(device, nn='cnn'):
    """
        Build a SPN (implemented as an einsum network). The structure is either
        the same as proposed in https://arxiv.org/pdf/1202.3732.pdf (referred to as
        poon-domingos) or a binary tree.

        In case of poon-domingos the image is split into smaller hypercubes (i.e. a set of
        neighbored pixels) where each pixel is a random variable. These hypercubes are split further
        until we operate on pixel-level. The spplitting is done randomly. For more information
        refer to the link above.
    """

    if config.structure == 'poon-domingos':
        pd_delta = [[config.height / d, config.width / d] for d in config.pd_num_pieces]
        graph = Graph.poon_domingos_structure(shape=(config.height, config.width), delta=pd_delta)
    elif config.structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=config.num_vars, depth=config.depth, num_repetitions=config.num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=config.num_vars,
            num_dims=config.num_dims,
            num_classes=1,
            num_sums=config.K,
            num_input_distributions=config.K,
            exponential_family=config.exponential_family,
            exponential_family_args=config.exponential_family_args,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)
    
    in_dim = (3*config.num_dims*config.num_vars) + 1000
    out_dims = [[config.num_vars, config.num_dims, 1, 3],
                [config.num_vars, config.num_dims, 1, 3], 
                [config.num_dims, config.num_dims, config.num_dims, 4],
                [config.num_dims, config.num_dims, 1, 2],
                [1, 1, 2]]
    print(f"Using NN={nn}")
    if nn == 'cnn':
        net = CNN(out_dims).to(device)
    elif nn == 'mlp':
        net = MLP(in_dim, out_dims, [256, 512]).to(device)
        for l in net.linear_layers:
            torch.nn.init.xavier_normal_(l.weight)
        for h in net.heads:
            torch.nn.init.xavier_normal_(h.weight)
    einet = EinsumNetwork.EinsumNetwork(graph, net, config.patch_size, args)
    einet.initialize()
    einet.to(device)
    return einet
   
def train(num_epochs, i, j, device, save_file):

    """
        Train ConFeinsum
    """
    torch.manual_seed(111)
    transform = Compose([ToTensor(), Resize(112), CenterCrop(112)])
    imagenet = ImageNet('/storage-01/datasets/imagenet/', transform=transform)
    clusters = np.load('/storage-01/ml-jseng/imagenet-clusters/vit_cluster_minibatch_10K.npy')
    img_ids = np.argwhere(clusters == 2400).flatten()
    subset = Subset(imagenet, img_ids)
    train_loader = DataLoader(subset, 64)
    rt = RTPT('JS', 'ConFEINSUM', num_epochs)
    rt.start()
    device = torch.device(f'cuda:{device}')
    einet = init_spn(device, nn='mlp')
    optim = torch.optim.Adam(einet.param_nn.parameters(), 0.002)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)

    for epoch in range(num_epochs):
        total_ll = 0.0
        for x, y in train_loader:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            
            ll = einet(x, y, i, j)
            ll.backward()

            #torch.nn.utils.clip_grad.clip_grad_norm_(einet.param_nn.parameters(), 1.)

            total_ll += ll

            optim.step()
        lr_schedule.step()
        print(f"Epoch {epoch} \t LL: {round(total_ll.item(), 3)} \t idx: [{i}, {j}]")
        rt.step()

    torch.save(einet.param_nn, save_file)

    return einet


def train_cluster(num_epochs, sqrt_num_patches):
    """
        Train sqrt_num_patches**2 Conditional Einsums
    """
    patch_ids = [[i, j] for i in range(1, sqrt_num_patches-1) for j in range(1, sqrt_num_patches-1)]
    print(f"Start Training of {len(patch_ids)} patches")
    patch_ids = np.array(patch_ids)
    num_procs = int(np.ceil(len(patch_ids) / config.num_processes))
    patch_ids = np.array_split(patch_ids, num_procs)

    for patch_batch in patch_ids:

        print(f"Train on {len(patch_batch)} in parallel")

        processes = []
        # we ignore the most outer patches to raise sample quality
        for i, j in patch_batch:
            print(f"Train on patch [{i}, {j}]")
            device = i % config.num_processes
            p = Process(target=train, args=(num_epochs, i, j, device, f'./models/model_{i}_{j}'))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()


if __name__ == '__main__':
    train_cluster(100, 14)