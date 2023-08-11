"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
from einsum.EinsumNetwork import EinsumNetwork, Args, log_likelihoods
from einsum.Graph import poon_domingos_structure, random_binary_trees
from torch.utils.data import Subset, DataLoader
from rtpt import RTPT
import config
import os
import torch
import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn
from datasets.datasets import Avazu, Income
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.EM import EM_optimization
import context

n_gpus = 1 if torch.cuda.is_available() else 0
@ray.remote(num_gpus=n_gpus)
class EinetNode:

    def __init__(self, dataset, chk_dir, group_id, num_epochs=3, rank=None) -> None:
        self.rank = rank
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.chk_dir = chk_dir
        self.group_id = group_id
        self.device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.einet = self._init_spn(self.device)
        self._load_data()
        self._rtpt = RTPT('JS', 'FedSPN', num_epochs)
        self._rtpt.start()

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
        """
        Training loop to train the SPN. Follows EM-procedure.
        """
        self.losses = []
        for epoch_count in range(self.num_epochs):
            self.einet.train()
            self._rtpt.step()
            total_ll = 0.0
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                outputs = self.einet.forward(x)
                ll_sample = log_likelihoods(outputs)
                log_likelihood = ll_sample.sum()
                log_likelihood.backward()
                self.einet.em_process_batch()
                total_ll += log_likelihood.detach().item()
            self.losses.append(total_ll)

            self.einet.em_update()
        return self.einet

    def assign_subset(self, train_inds, test_inds):
        """
            assign subset to this client
        """
        self.train_data = Subset(self.train_data, train_inds)
        self.test_data = Subset(self.test_data, test_inds)
        self.train_loader = DataLoader(self.train_data, batch_size=32)
        self.test_loader = DataLoader(self.test_data, batch_size=32)

    def _load_data(self):
        """
            Load data
        """
        self.train_data, self.test_data = get_corel5k_data()

    def get_losses(self):
        return self.losses


    def _init_spn(self, device):
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

    def _train(self, einet, train_loader, num_epochs, device, chk_path, mean=None, save_model=True):

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
    
@ray.remote
class FlowNode:

    def __init__(self, dataset, group_id, rank=None) -> None:
        self.rank = rank
        self.dataset = dataset
        self.group_id = group_id
        self._load_data()
        self._rtpt = RTPT('JS', 'FedSPN', 1)
        self._rtpt.start()
        self.feature_index = []
        self.subspaces = []
        self.spns = {}

    def get_group_id(self):
        return self.group_id
    
    def _query(self, spn, query):
        """
            Query SPN
        """
        q = np.zeros(config.num_vars)
        for i, v in query.items():
            q[i] = v
        
        if len(query) == config.num_vars:
            # all RVs are set to some value, get joint
            ll = log_likelihood(spn, q.reshape(1, -1))
            return ll
        else:
            # some RVs are not set, get marginal
            keep = list(query.keys())
            marg_spn = marginalize(spn, keep)
            return log_likelihood(marg_spn, q)
        
    def query(self, subspace, query):
        spn = self.spns[tuple(subspace)]
        return self._query(spn, query)

    def train(self):
        for subspace in self.subspaces:
            ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
            ctxt.add_domains(self.train_data[:, subspace])
            spn = learn_mspn(self.train_data[:, subspace], ctxt)
            EM_optimization(spn, self.train_data[:, subspace])
            self.spns[tuple(subspace)] = spn

    def assign_subset(self, train_inds):
        """
            assign subset to this client
        """
        self.X = self.train_data.features[train_inds, :].numpy()
        self.y = self.train_data.targets[train_inds].numpy()
        self.train_data = np.hstack((self.X, self.y.reshape(-1, 1)))

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """

    def get_dataset_len(self):
        return len(self.train_data)

    def _load_data(self):
        """
            Load data
        """
        if self.dataset == 'avazu':
            self.train_data = Avazu('../../datasets/avazu/', split='train')
        elif self.dataset == 'income':
            self.train_data = Income('../../datasets/income/', split='train')
        else:
            raise ValueError(f'No such dataset: {self.dataset}')
        
    def get_feature_space(self):
        return self.feature_index
    
    def add_subspace(self, subspace):
        self.subspaces.append(subspace)
