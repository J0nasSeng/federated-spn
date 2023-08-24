"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
#from einet.einet import Einet, EinetConfig
from torch.utils.data import DataLoader
from rtpt import RTPT
import torch
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.algorithms.EM import EM_optimization
import context
import utils
from einet.einet import Einet, EinetConfig
from einet.distributions.normal import RatNormal

n_gpus = 1 if torch.cuda.is_available() else 0
@ray.remote(num_gpus=n_gpus)
class EinetNode:

    def __init__(self, dataset, num_epochs=10, num_classes=2) -> None:
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.einets = {}
        self.subspaces = []
        self._rtpt = RTPT('JS', 'FedSPN', num_epochs)
        self._rtpt.start()
        self.num_classes = num_classes

    def train(self):
        """
            Train SPN on local data
            Training loop to train the SPN. Follows EM-procedure.
        """
        for subspace, train_loader in self.subspaces:
            einet_cfg = EinetConfig(len(subspace) - 1, num_classes=self.num_classes, 
                                    leaf_type=RatNormal, leaf_kwargs={}, depth=5, num_leaves=20,
                                    num_sums=20, num_repetitions=10)
            einet = Einet(einet_cfg)
            optim = torch.optim.SGD(einet.parameters(), 0.1)
            cross_entropy = torch.nn.CrossEntropyLoss()
            self.losses = []
            for epoch_count in range(self.num_epochs):
                optim.zero_grad()
                self._rtpt.step()
                total_ll = 0.0
                for i, (x, y) in enumerate(train_loader):
                    x = x.to(device=self.device, dtype=torch.float32)
                    y = y.to(device=self.device, dtype=torch.long)
                    x = x.unsqueeze(1)
                    outputs = einet(x)
                    loss = cross_entropy(outputs, y)
                    
                    loss.backward()
                    optim.step()
                    total_ll += loss.item()
                self.losses.append(total_ll / len(train_loader))
                print(f"Epoch {epoch_count+1}/{self.num_epochs}: \t {total_ll / len(train_loader)}")
            self.einets[tuple(subspace)] = einet

    def assign_subset(self, subspace_with_data):
        self.subspaces.append(subspace_with_data)

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """

    def get_dataset_len(self):
        data_len = 0
        for _, loader in self.subspaces:
            data_len += len(loader) * loader.batch_size
        return data_len

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
            spn = utils.adjust_scope(spn, subspace)
            self.spns[tuple(subspace)] = spn

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """

    def get_dataset_len(self):
        len_data = sum(len(data) for _, data in self.subspaces)
        return len_data
    
    def get_spn(self, subspace):
        return self.spns[subspace]
    
    def get_spns(self):
        return self.spns
    
    def assign_subset(self, subspace_with_data):
        self.subspaces.append(subspace_with_data)

