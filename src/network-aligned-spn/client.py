"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
#from einet.einet import Einet, EinetConfig
from torch.utils.data import DataLoader, Subset
from rtpt import RTPT
import torch
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context, Sum
from spn.algorithms.EM import EM_optimization
import context
import utils
from einet.einet import Einet, EinetConfig
from einet.distributions.normal import RatNormal
from einsum.EinsumNetwork import Args, EinsumNetwork
from einsum.Graph import random_binary_trees
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from spn.algorithms.Inference import log_likelihood
import normflows as nf
import torchvision as tv
from torchvision.transforms.functional import crop
from det.tree import DensityTree

n_gpus = 0.25 if torch.cuda.is_available() else 0
@ray.remote(num_gpus=n_gpus)
class EinetNode:

    def __init__(self, dataset, num_epochs=20, num_classes=2, train_algo='em', device=-1) -> None:
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.device = torch.device(f'cuda:0') if device > -1 else torch.device('cpu')
        self.losses = []
        self.einets = {}
        self.subspaces = []
        self._rtpt = RTPT('JS', 'FedSPN', num_epochs)
        self._rtpt.start()
        self.num_classes = num_classes
        self.train_algo = train_algo

    def train(self):
        """
            Train SPN on local data
            Training loop to train the SPN. Follows SGD.
        """
        if self.train_algo == 'em':
            self._train_em()
        elif self.train_algo == 'sgd':
            self._train_sgd()

    def _train_sgd(self):
        self.config = EinetConfig(len(self.subspaces[0][0]), num_classes=self.num_classes, 
                                    leaf_type=RatNormal, leaf_kwargs={}, depth=4, num_leaves=20,
                                    num_sums=20, num_repetitions=10)
        for subspace, train_loader in self.subspaces:
            einet = Einet(self.config).to(self.device)
            optim = torch.optim.SGD(einet.parameters(), 0.001)
            for epoch_count in range(self.num_epochs):
                optim.zero_grad()
                self._rtpt.step()
                total_ll = 0.0
                for i, (x, y) in enumerate(train_loader):
                    x = x.to(device=self.device, dtype=torch.float32)
                    y = y.to(device=self.device, dtype=torch.float32)
                    x_in = torch.cat((x, y.unsqueeze(1)), dim=1)
                    outputs = einet(x_in)
                    loss = torch.mean(utils.log_likelihoods(outputs))

                    loss.backward()
                    optim.step()
                    total_ll += loss.item()
                    if i % 50 == 0:
                        print(f"Epoch {epoch_count+1}/{self.num_epochs}: Batch: {i}/{len(train_loader)} \t {total_ll / len(train_loader)}")
                self.losses.append(total_ll / len(train_loader))
                print(f"Epoch {epoch_count+1}/{self.num_epochs}: \t {total_ll / len(train_loader)}")
            self.einets[tuple(subspace)] = einet

    def _train_em(self):
        self.config = Args(
                len(self.subspaces[0][0]),
                1,
                num_input_distributions=40,
                num_sums=40,
                num_classes=self.num_classes,
                exponential_family_args={
                    'min_var': 1e-6,
                    'max_var': 1.
                },
                online_em_frequency=5,
                online_em_stepsize=0.1
            )
        for subspace, train_loader in self.subspaces:
            graph = random_binary_trees(len(subspace), 4, 10)
            einet = EinsumNetwork(graph, self.config)
            einet.config = self.config # make compatible with FC framework
            einet.initialize()
            einet = einet.to(self.device)
            for epoch_count in range(self.num_epochs):

                total_ll = 0.0
                for i, (x, y) in enumerate(train_loader):

                    x = x.to(device=self.device, dtype=torch.float32)
                    y = y.to(device=self.device, dtype=torch.float32)
                    x_in = torch.cat((x, y.unsqueeze(1)), dim=1)
                    x_in = x_in.unsqueeze(2)
                    x_in = x_in.to(self.device)
                    ll_sample = einet.forward(x_in)
                    log_likelihood = ll_sample.mean()
                    log_likelihood.backward()
                    einet.em_process_batch()
                    total_ll += log_likelihood.item()

                self.losses.append(total_ll / len(train_loader))
                einet.em_update()
                print(f"Epoch {epoch_count+1}/{self.num_epochs}: \t {total_ll / len(train_loader)}")
            self.einets[tuple(subspace)] = einet.to('cpu')


    def assign_subset(self, subspace_with_data):
        self.subspaces.append(subspace_with_data)

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """
        pass

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
class NormalizingFlowNode:

    def __init__(self, setting, device) -> None:
        self.setting = setting
        self.device = device
        self.subspace = None

    def _create_flow(self):
        # Define flows
        L = 3
        K = 16
        torch.manual_seed(0)

        # TODO: adapt according to client's feature space
        input_shape = (3, 32, 32)
        channels = 3
        hidden_channels = 256
        split_mode = 'channel'
        scale = True
        num_classes = 10

        # Set up flows, distributions and merge operations
        q0 = []
        merges = []
        flows = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                            split_mode=split_mode, scale=scale)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                                input_shape[2] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                                input_shape[2] // 2 ** L)
            q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]


        # Construct flow model with the multiscale architecture
        self.model = nf.MultiscaleFlow(q0, flows, merges).to(self.device)

    def train(self):
        assert self.subspace is not None, 'subspace must be set before training'
        sample_idx, x_subspace, y_subspace = self.subspace
        x1, x2 = x_subspace
        y1, y2 = y_subspace
        batch_size = 128

        transform = tv.transforms.Compose([tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.)])
        train_data = tv.datasets.ImageNet('datasets/', train=True,
                                        download=False, transform=transform)
        train_data = Subset(train_data, sample_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                drop_last=True)

        train_iter = iter(train_loader)

        max_iter = 20000

        loss_hist = np.array([])

        optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        for i in range(max_iter):
            # TODO: cut data s.t. model only learns on feature subspace of client
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            optimizer.zero_grad()
            if x1 is not None:
                x = crop(x, x1, y1, x2-x1, y2-y1)
            loss = self.model.forward_kld(x.to(self.device), y.to(self.device))
                
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    def assign_subset(self, subspace):
        self.subspace = subspace

@ray.remote
class DensityTreeNode:

    def __init__(self, dataset, num_clusters=2, max_depth=4) -> None:
        self.dataset = dataset
        self.models = {}
        self.subspaces = []
        self.num_clusters = num_clusters
        self.max_depth = max_depth
        self.data: np.ndarray
        self.feature_types = []
    
    def train(self):
        for subspace, feature_types in zip(self.subspaces, self.feature_types):
            if self.num_clusters == 1:
                model = self.train_single(subspace, feature_types)
            else:
                model = self.train_cluster(subspace, feature_types)
            self.models[tuple(subspace)] = model


    def train_single(self, subspace, fts):
        model = DensityTree(8, fts, min_leaf_instances=10, leaf_type='hist', 
                            num_bins=2, scope=subspace)
        model.train(self.data[:, subspace])
        return [model]

    def train_cluster(self, subspace, fts):
        kmeans = KMeans(self.num_clusters)
        clusters = kmeans.fit_predict(self.data[:, subspace]).flatten()
        models = []
        for c in np.unique(clusters):
            idx = np.argwhere(clusters == c).flatten()
            subset = self.data[idx]
            tree = DensityTree(8, fts, min_leaf_instances=10, leaf_type='hist', 
                               num_bins=3, scope=subspace)
            tree.train(subset[:, subspace])
            models.append(tree)
        return models
        
    def get_dataset_len(self):
        return len(self.data)
    
    def get_model(self, subspace):
        return self.models[subspace]
    
    def get_models(self):
        return self.models
    
    def assign_subset(self, data):
        self.data = data

    def assign_feature_spaces(self, feature_spaces):
        self.subspaces = feature_spaces

    def assign_feature_types(self, feature_types):
        self.feature_types = feature_types

@ray.remote
class RandomForestNode:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.subspace = []
        self.model = None

    def train(self):
        self.model = RandomForestClassifier(100)
        _, train_data = self.subspace
        X, y = train_data[:, :-1], train_data[:, -1]
        self.model.fit(X, y)


    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """
        return self.subspace[0]

    def get_dataset_len(self):
        return len(self.subspace[1])
    
    def get_forest(self):
        return self.model
    
    def assign_subset(self, subspace_with_data):
        self.subspace = subspace_with_data

@ray.remote
class FlowNode:

    def __init__(self, dataset, spn_structure, num_clusters, setting, glueing, cluster_by_label) -> None:
        self.dataset = dataset
        self.spn_structure = spn_structure
        self.num_clusters = num_clusters
        self.setting = setting
        self.glueing = glueing
        self.cluster_by_label = cluster_by_label
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
        if self.cluster_by_label == 1:
            labels = np.unique(train_data[:, -1])
            cluster_spns = []
            for l in labels:
                idx = np.argwhere(train_data[:, -1].flatten() == l).flatten()
                subset = train_data[idx]
                ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
                ctxt.add_domains(subset)
                spn = learn_mspn(subset, ctxt, min_instances_slice=200, threshold=0.3)
                spn = utils.adjust_scope(spn, subspace)
                cluster_spns.append(spn)
            spn = self._build_cluster_mixture(cluster_spns, train_data[:, -1].flatten())
            self.spns[tuple(subspace)] = [spn]
        else:
            if self.num_clusters > 1:
                kmeans = KMeans(self.num_clusters)
                clusters = kmeans.fit_predict(train_data)
                cluster_spns = []
                for c in np.unique(clusters):
                    idx = np.argwhere(clusters == c).flatten()
                    subset = train_data[idx]
                    node_types = utils.infer_node_type(subset, 15)
                    types = [t for t, _ in node_types]
                    ctxt = Context(meta_types=[context.ctxts[self.dataset][i] for i in subspace])
                    #ctxt = Context(parametric_types=types)
                    ctxt.add_domains(subset)
                    spn = learn_mspn(subset, ctxt, min_instances_slice=100, threshold=0.4)
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
                spn = learn_mspn(train_data, ctxt, min_instances_slice=50, threshold=0.4)
                spn = utils.adjust_scope(spn, subspace)
                self.spns[tuple(subspace)] = [spn]
            

    def _train_rat(self, subspace, train_data):
        if self.cluster_by_label == 1:
            labels = np.unique(train_data[:, -1])
            cluster_spns = []
            for l in labels:
                idx = np.argwhere(train_data[:, -1].flatten() == l).flatten()
                subset = train_data[idx]
                spn = self._build_rat_spn(subspace, subset)
                EM_optimization(spn, subset)
                spn = utils.adjust_scope(spn, subspace)
                cluster_spns.append(spn)
            spn = self._build_cluster_mixture(cluster_spns, train_data[:, -1].flatten())
            self.spns[tuple(subspace)] = [spn]
        else:
            if self.num_clusters > 1:
                kmeans = KMeans(self.num_clusters)
                clusters = kmeans.fit_predict(train_data)
                cluster_spns = []
                for c in np.unique(clusters):
                    idx = np.argwhere(clusters == c).flatten()
                    subset = train_data[idx]
                    spn = self._build_rat_spn(subspace, train_data)
                    EM_optimization(spn, subset)
                    spn = utils.adjust_scope(spn, subspace)
                    cluster_spns.append(spn)
                if self.setting == 'horizontal' or self.glueing == 'naive':
                    spn = self._build_cluster_mixture(cluster_spns, clusters)
                    self.spns[tuple(subspace)] = [spn]
                else:
                    self.spns[tuple(subspace)] = cluster_spns
            else:
                spn = self._build_rat_spn(subspace, train_data)
                EM_optimization(spn, train_data)
                spn = utils.adjust_scope(spn, subspace)
                self.spns[tuple(subspace)] = [spn]

    def _build_cluster_mixture(self, spns, clusters):
        assert len(spns) == len(np.unique(clusters))
        root = Sum()
        weights = []
        scopes = []
        for s in spns:
            scopes += list(s.scope)
        root.scope = list(set(scopes))
        for c in np.unique(clusters):
            w = np.argwhere(c == clusters).sum()
            weights.append(w)
        root.weights = np.array(weights)
        root.weights = root.weights / np.sum(root.weights)
        root.children = spns
        root = utils.reassign_node_ids(root)
        return root


    def _build_rat_spn(self, subspace, train_data):
        region_graph = utils.random_region_graph(0, list(range(train_data.shape[1])), [])
        node_types = utils.infer_node_type(train_data, 60)
        dists = {i: nt for i, nt in enumerate(node_types)}
        curr_layer = [n for n in region_graph.nodes if len(list(region_graph.pred[n])) == 0]
        spn = utils.region_graph_to_spn(region_graph, curr_layer, dists)
        spn = utils.reassign_node_ids(spn)
        return spn

    def get_feature_ids(self):
        """
            Retrieve feature ids hold by the client
        """
        pass

    def get_dataset_len(self):
        len_data = sum(len(data) for _, data in self.subspaces)
        return len_data
    
    def get_spn(self, subspace):
        return self.spns[subspace]
    
    def get_spns(self):
        return self.spns
    
    def assign_subset(self, subspace_with_data):
        self.subspaces.append(subspace_with_data)