"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
from einet.einet import Einet, EinetConfig
from torch.utils.data import DataLoader
from rtpt import RTPT
import torch
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.algorithms.EM import EM_optimization
import context

n_gpus = 1 if torch.cuda.is_available() else 0
@ray.remote(num_gpus=n_gpus)
class EinetNode:

    def __init__(self, dataset, num_epochs=10) -> None:
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.einets = {}
        self.subspaces = []
        self._rtpt = RTPT('JS', 'FedSPN', num_epochs)
        self._rtpt.start()

    def train(self):
        """
            Train SPN on local data
        """
        """
        Training loop to train the SPN. Follows EM-procedure.
        """
        for subspace in self.subspaces:
            einet_cfg = EinetConfig(len(subspace))
            einet = Einet(einet_cfg)
            optim = torch.optim.Adam(self.einet.parameters(), 1.0)
            cross_entropy = torch.nn.CrossEntropyLoss()
            self.losses = []
            for epoch_count in range(self.num_epochs):
                optim.zero_grad()
                self._rtpt.step()
                total_ll = 0.0
                for i, (x, y) in enumerate(self.train_loader):
                    x = x.to(self.device)
                    outputs = einet(x)
                    loss = cross_entropy(outputs, y)
                    
                    loss.backward()
                    optim.step()
                    total_ll += loss.item()
                self.losses.append(total_ll)
            self.einets[subspace] = einet

    def assign_subset(self, train_data):
        """
            assign subset to this client
        """
        self.train_data = train_data
        self.train_loader = DataLoader(self.train_data, 32)

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """

    def get_dataset_len(self):
        return len(self.train_data)
    
    def assign_subspace(self, subspace):
        self.subspaces.append(subspace)

    def get_spn(self, subspace):
        return self.einets[subspace]
    
    def get_spns(self):
        return self.einets

@ray.remote
class FlowNode:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self._rtpt = RTPT('JS', 'FedSPN', 1)
        self._rtpt.start()
        self.subspaces = []
        self.spns = {}

    def train(self):
        for subspace, train_data in self.subspaces:
            ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
            ctxt.add_domains(train_data)
            spn = learn_mspn(train_data, ctxt)
            EM_optimization(spn, train_data, iterations=3)
            self.spns[tuple(subspace)] = spn

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """

    def get_dataset_len(self):
        return len(self.train_data)
    
    def get_spn(self, subspace):
        return self.spns[subspace]
    
    def get_spns(self):
        return self.spns
    
    def assign_subset(self, subspace_with_data):
        self.subspaces.append(subspace_with_data)

