from nns import MLP
import numpy as np
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from einsum import EinsumNetwork, Graph, EinetMixture
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

def init_spn(device):
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
    elif config.structure == 'flat-binary-tree':
        graph = Graph.binary_tree_spn(shape=(config.height, config.width))
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

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    return einet

def train(einet: EinsumNetwork.EinsumNetwork, train_loader, num_epochs, i, j, device, chk_path, save_model=True):

    """
        Train ConFeinsum
    """
    parameters = list(einet.parameters())
    in_dim = 3*3*64 + 14
    mlp = MLP(in_dim, len(parameters))

    optim = torch.optim.Adam(mlp.parameters(), 0.01)

    for _ in range(num_epochs):
        for x, y in train_loader:
            optim.zero_grad()
            x = x.to(device)
            # patch images and obtain 8x8 patches
            patches = extract_image_patches(x, 8, 8)
            x_in = patches[:, :, i, j]
            x_prev = get_surrounding_patches(patches, i, j)
            # flatten all previous patches and concat along feature dimension for NN
            x_prev = [x.squeeze().reshape(x_in.shape[0], -1) for x in x_prev]
            x_prev = torch.cat(x_prev, dim=1)
            y_oh = F.one_hot(y)
            x_prev = torch.cat([x_prev, y_oh], dim=1)

            einet_params = mlp(x_prev)
            
            total_ll = 0.
            x_in = x_in.permute((0, 2, 3, 1))
            x_in = x_in.reshape(x_in.shape[0], config.num_vars, config.num_dims)

            # set einet parameters for each batch as parameters differ
            # for each sample
            for i, einet_param in enumerate(einet_params):
                einet = set_einet_weights(einet, einet_param)

                # evaluate log-likelihood
                total_ll += einet(x_in[i])

            total_ll.backward()

            optim.step()
   
