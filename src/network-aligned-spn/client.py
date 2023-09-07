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
from spn.structure.Base import Context, Sum
from spn.algorithms.EM import EM_optimization
import context
import utils
from einet.einet import Einet, EinetConfig
from einet.distributions.normal import RatNormal
from sklearn.cluster import KMeans
import numpy as np

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

    def __init__(self, dataset, spn_structure, num_clusters, setting, glueing) -> None:
        self.dataset = dataset
        self.spn_structure = spn_structure
        self.num_clusters = num_clusters
        self.setting = setting
        self.glueing = glueing
        self._rtpt = RTPT('JS', 'FedSPN', 1)
        self._rtpt.start()
        self.subspaces = []
        self.spns = {}

    def train(self):
        for subspace, train_data in self.subspaces:
            if self.spn_structure == 'learned':
                self._train_learned(subspace, train_data)
            elif self.spn_structure == 'rat':
                self._train_rat(subspace, train_data)

    def _train_learned(self, subspace, train_data):
        if self.num_clusters > 1:
            kmeans = KMeans(self.num_clusters)
            clusters = kmeans.fit_predict(train_data)
            cluster_spns = []
            for c in np.unique(clusters):
                idx = np.argwhere(clusters == c)
                subset = train_data[idx]
                ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
                ctxt.add_domains(subset)
                spn = learn_mspn(subset, ctxt)
                spn = utils.adjust_scope(spn, subspace)
                cluster_spns.append(spn)
            if self.setting == 'horizontal' or self.glueing == 'naive':
                spn = self._build_cluster_mixture(cluster_spns, clusters)
                self.spns[tuple(subspace)] = [spn]
            else:
                self.spns[tuple(subspace)] = cluster_spns
        else:
            ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
            ctxt.add_domains(train_data)
            spn = learn_mspn(train_data, ctxt)
            spn = utils.adjust_scope(spn, subspace)
            self.spns[tuple(subspace)] = [spn]
            

    def _train_rat(self, subspace, train_data):
        if self.num_clusters > 1:
            kmeans = KMeans(self.num_clusters)
            clusters = kmeans.fit_predict(train_data)
            cluster_spns = []
            for c in np.unique(clusters):
                idx = np.argwhere(clusters == c)
                subset = train_data[idx]
                spn = self._build_rat_spn(subspace, train_data.shape[1])
                EM_optimization(spn, subset)
                spn = utils.adjust_scope(spn, subspace)
                cluster_spns.append(spn)
            if self.setting == 'horizontal' or self.glueing == 'naive':
                spn = self._build_cluster_mixture(cluster_spns, clusters)
                self.spns[tuple(subspace)] = [spn]
            else:
                self.spns[tuple(subspace)] = cluster_spns
        else:
            spn = self._build_rat_spn(subspace, train_data.shape[1])
            EM_optimization(spn, train_data)
            spn = utils.adjust_scope(spn, subspace)
            self.spns[tuple(subspace)] = [spn]

    def _build_cluster_mixture(spns, clusters):
        assert len(spns) == len(np.unique(clusters))
        root = Sum()
        weights = []
        scopes = []
        for s in spns:
            scopes += list(s.scope)
        root.scope = list(set(scopes))
        for c in np.unique(clusters):
            w = np.argwhere(c == clusters).sum() / len(clusters)
            weights.append(w)
        root.weights = np.array(weights)
        root = utils.reassign_node_ids(root)
        return root


    def _build_rat_spn(self, subspace, num_vars):
        region_graph = utils.random_region_graph(0, list(range(num_vars)), [])
        dists = {i: context.node_types[i] for i in subspace}
        curr_layer = [n for n in region_graph.nodes if len(list(region_graph.pred[n])) == 0]
        spn = utils.region_graph_to_spn(region_graph, curr_layer, dists)
        spn = utils.reassign_node_ids(spn)
        return spn

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