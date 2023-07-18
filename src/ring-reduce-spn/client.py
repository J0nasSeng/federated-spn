"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
from einsum.EinsumNetwork import EinsumNetwork, Args, log_likelihoods
from einsum.Graph import poon_domingos_structure, random_binary_trees
from torch.utils.data import Subset, DataLoader
from datasets import get_medical_data
import config
import json
import os
import torch
import numpy as np

@ray.remote(num_gpus=1)
class Node:

    def __init__(self, dataset, chk_dir, group_id, num_epochs=3, rank=None) -> None:
        self.rank = rank
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.chk_dir = chk_dir
        self.group_id = group_id
        self.device = torch.device(f'cuda')
        self.einet = init_spn(self.device)
        self._load_data()

    def get_group_id(self):
        return self.group_id

    def query(self, query):
        """
            Query SPN
        """
        q = torch.zeros(config.num_vars)
        for i, v in query.items():
            q[i] = v
        q = q.unsqueeze(0).to(self.device)
        
        if len(query) == config.num_vars:
            # all RVs are set to some value, get joint
            x = self.einet.forward(q)
            return log_likelihoods(x).detach().squeeze().item()
        else:
            # some RVs are not set, get marginal
            ref = np.arange(config.num_vars)
            marg_idx = [i for i in ref if i not in list(query.keys())]
            self.einet.set_marginalization_idx(marg_idx)
            x = self.einet.forward(q)
            return log_likelihoods(x).detach().squeeze().item()

    def train(self, return_spn=True):
        """
            Train SPN on local data
        """
        self.trained = False
        self.einet, self.losses = train(self.einet, self.train_loader, self.num_epochs, self.device, self.chk_dir)
        self.trained = True
        if return_spn:
            return self.einet

    def assign_subset(self, train_inds, test_inds):
        """
            assign subset to this client
        """
        self.train_data = Subset(self.train_data, ray.get(train_inds))
        self.test_data = Subset(self.test_data, ray.get(test_inds))
        self.train_loader = DataLoader(self.train_data, batch_size=32)
        self.test_loader = DataLoader(self.test_data, batch_size=32)

    def _load_data(self):
        """
            Load data
        """
        self.train_data, self.test_data = get_medical_data()

    def get_losses(self):
        return self.trained


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
        graph = poon_domingos_structure(shape=(config.height, config.width), delta=[4], axes=[1])
    elif config.structure == 'binary-trees':
        graph = random_binary_trees(num_var=config.num_vars, depth=config.depth, num_repetitions=config.num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = Args(
            num_var=config.num_vars,
            num_dims=config.num_dims,
            num_classes=1,
            num_sums=config.K,
            num_input_distributions=config.K,
            exponential_family=config.exponential_family,
            exponential_family_args=config.exponential_family_args,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)

    einet = EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    return einet

def make_subset(dataset, split, rank, inds_file):
    with open(inds_file) as f:
        inds_dict = json.load(f)
    indices = inds_dict[rank][split]
    subset = Subset(dataset, indices)
    return subset

def train(einet, train_loader, num_epochs, device, chk_path, mean=None, save_model=True):

    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    losses = []
    for epoch_count in range(num_epochs):
        einet.train()

        if save_model and (epoch_count > 0 and epoch_count % config.checkpoint_freq == 0):
            torch.save(einet, os.path.join(chk_path, f'chk_{epoch_count}.pt'))

        total_ll = 0.0
        for i, (x, y) in enumerate(train_loader):
            print(x)
            x = x.to(device)
            outputs = einet.forward(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()
        losses.append(total_ll)

        einet.em_update()
    return einet, losses