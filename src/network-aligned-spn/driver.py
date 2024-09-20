"""
    This file starts the driver process of a ray cluster.
    It then coordinates training of the clients. 
    Note that this just simulates the ring-reduce network topology
    proposed in our FedSPN paper.
    Nevertheless, semantically it's performing the same operations
    as ring reduce algorithm.
"""

import ray
import numpy as np
from client import FlowNode, EinetNode, RandomForestNode, DensityTreeNode
from datasets.utils import get_horizontal_train_data, get_test_data, get_vertical_train_data, get_hybrid_train_data, make_data_loader
from spn.structure.Base import Sum, Product, get_nodes_by_type
from spn.algorithms.MPE import mpe
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.EM import EM_optimization
from scipy.special import softmax
from optim import add_node_em_update, cond_sum_em_update, EM_optimization_network
from rtpt import RTPT
import logging
import sys
import warnings
import argparse
from sklearn.metrics import accuracy_score, f1_score
import utils
from spn_leaf import SPNLeaf, DensityLeaf
from einet.layers import Sum as SumLayer
import torch
import torch.nn as nn
import pandas as pd
import os
import context as ctxt
warnings.filterwarnings('ignore')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

rtpt = RTPT('JS', 'FedSPN Driver', 3)
rtpt.start()

class SPFlowServer:

    def train(self, train_data, feature_spaces, args):
        """
            This function starts a local ray cluster, splits data into equal sized
            subsets and trains one client/worker on each subset.
            Each client creates its own local SPN.
        """
        ray.init()
        train_jobs = []
        assign_jobs = []
        nodes = []
        for c in range(args.num_clients):
            logging.info(f'Train node {c}')
            node = FlowNode.remote(args.dataset, args.structure, args.num_clusters,
                                   args.setting, args.glueing, args.cluster_by_label)
            nodes.append(node)
            train_subset = train_data[c]
            subspace = feature_spaces[c]
            assign_jobs.append(node.assign_subset.remote((subspace, train_subset)))
        ray.get(assign_jobs)
            
        rtpt.step()

        for c in range(args.num_clients):
            train_jobs.append(nodes[c].train.remote())
        
        ray.get(train_jobs)
        rtpt.step()

        return nodes

    def classify(self, spn, test_data):
        """
            Classify samples using spn
        """
        test_data[:, -1] = np.nan
        pred = mpe(spn, test_data)
        return pred[:, -1].flatten()

    def build_spn_horizontal(self, nodes):
        """
            Collect all SPNs residing on clients and introduce a new
            root (sum node), weighted by dataset size on each client
        """
        # leafs = [SPNLeaf(c) for c in range(config.num_clients)]
        leaf_dict = [ray.get(node.get_spns.remote()) for node in nodes]
        leafs = [list(l.values())[0][0] for l in leaf_dict]
        ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
        norm = sum(ds_len)
        weights = [d / norm for d in ds_len]
        spn = Sum(weights, leafs)
        spn.scope = []
        for c in spn.children:
            spn.scope = list(set(spn.scope).union(set(c.scope)))
        spn = utils.reassign_node_ids(spn)
        return spn
    
    def build_spn_verhyb_naive(self, feature_subspaces, nodes):
        """
            Naively glue together client SPNs in vertical and hybrid setting.
            Each client holds one SPN.
            In hybrid setting, first the client SPNs which share the same feature
            space (scope) are connected by a mixture node.

            Then same as vertical case in which one Prodcut node is introduced
            which connects all client SPNs into one SPN.
        """
        spn = Product()
        added_nodes = [spn]
        for clients, subspace in feature_subspaces.items():
            if len(clients) > 1:
                s = Sum()
                leafs = []
                for c in clients:
                    # this yields an array with exactly one SPN included
                    leafs.append(ray.get(nodes[c].get_spn.remote(tuple(subspace)))[0])
                s.children = leafs
                s.weights = np.repeat(1/len(s.children), len(s.children))
                s.scope = set().union(*[set(l.scope) for l in leafs])
                spn.children += [s]
                spn.scope = set().union(*[c.scope for c in spn.children])
                added_nodes.append(s)
            else:
                node_idx = list(clients)[0]
                client_spn = ray.get(nodes[node_idx].get_spn.remote(tuple(subspace)))[0]
                spn.children += [client_spn]
                spn.scope = set().union(spn.scope, client_spn.scope)
        spn = utils.reassign_node_ids(spn)
        return spn, added_nodes
    
    def build_spn_verhyb_combinatorial(self, feature_subspaces, nodes):
        """
            Glue together client SPNs in vertical and hybrid setting.
            Each client holds N SPNs, each corresponding to one cluster.
            In hybrid setting, first the client SPNs which share the same feature
            space (scope) are put together in N mixtures, resulting in N new SPNs.

            Then same as vertical case in which one Prodcut node is introduced
            for each combinaion of clusters, followed by a mixture (root node)
            weighting the "combinatorial clusters".
        """
        leafs = []
        added_nodes = []
        for clients, subspace in feature_subspaces.items():
            if len(clients) > 1:
                # in hybrid case multple clients can hold same subspace
                client_spns = []
                for c in clients:
                 # build mixture over common subspaces
                    spns = ray.get(nodes[c].get_spn.remote(tuple(subspace)))
                    client_spns.append(spns)
                
                num_spns = len(client_spns[0]) # number is same for all clients
                joint_leafs = []
                for i in range(num_spns):
                    spns = [cs[i] for cs in client_spns]
                    sum = Sum()
                    sum.children = spns
                    sum.weights = np.repeat(1 / len(spns), spns)
                    scopes = []
                    for s in spns:
                        scopes += list(s.scope)
                    sum.scope = list(set(scopes))
                    sum = utils.reassign_node_ids(sum)
                    joint_leafs.append(sum)
                added_nodes += joint_leafs
                leafs.append(joint_leafs)

            else:
                node_idx = list(clients)[0]
                client_spns = ray.get(nodes[node_idx].get_spn.remote(tuple(subspace)))
                leafs.append(client_spns)
        
        fedspn = utils.build_fedspn_head(leafs)
        added_nodes = [fedspn] + added_nodes
        return fedspn, added_nodes

    def build_spn(self, feature_subspaces, nodes, args):
        """
            Construct FedSPN based on setting (horizontal, vertical or hybrid)
        """
        if len(feature_subspaces) == 1:
            # horizontal case
            spn = self.build_spn_horizontal(nodes)
            return spn, None
        else:
            # hybrid & vertical case
            if args.glueing == 'naive':
                spn, added_nodes = self.build_spn_verhyb_naive(feature_subspaces, nodes)
            elif args.glueing == 'combinatorial':
                spn, added_nodes = self.build_spn_verhyb_combinatorial(feature_subspaces, nodes)
            return spn, added_nodes
        
class RandomForestServer:

    def train(self, train_data, feature_spaces, args):
        """
            This function starts a local ray cluster, splits data into equal sized
            subsets and trains one client/worker on each subset.
            Each client creates its own local random forest.
        """
        ray.init()
        train_jobs = []
        assign_jobs = []
        nodes = []
        for c in range(args.num_clients):
            logging.info(f'Train node {c}')
            node = RandomForestNode.remote(args.dataset)
            nodes.append(node)
            train_subset = train_data[c]
            subspace = feature_spaces[c]
            assign_jobs.append(node.assign_subset.remote((subspace, train_subset)))
        ray.get(assign_jobs)
        
        rtpt.step()

        for c in range(args.num_clients):
            train_jobs.append(nodes[c].train.remote())
        
        ray.get(train_jobs)
        rtpt.step()

        return nodes
    
    def build_fc(self, nodes):
        self.fc = []
        for client in nodes:
            model = ray.get(client.get_forest.remote())
            ds_len = ray.get(client.get_dataset_len.remote())
            subspace = ray.get(client.get_feature_ids.remote())
            self.fc.append([model, ds_len, subspace])

        ds_lens = [dsl for _, dsl, _ in self.fc]
        weights = np.array(ds_lens) / np.sum(ds_lens)
        for i in range(len(weights)):
            self.fc[i][1] = weights[i]

    def classify(self, test_data):
        client_probas = []
        weights = []
        for model, weight, subspace in self.fc:
            test_subset = test_data[:, subspace]
            probs = model.predict_proba(test_subset) # dimensions n x d_c
            client_probas.append(probs)
            weights.append(weight)
        
        client_probas = np.array(client_probas) # dimensions c x n x k
        # compute mixture probabilities for each sample using einsum
        client_probas = torch.from_numpy(client_probas)
        weights = torch.from_numpy(np.array(weights))
        probas = torch.einsum('cnd,c->nd', client_probas, weights)
        return probas.numpy()

class DenistyTreeServer:

    def train(self, train_data, feature_spaces, feature_types, args, **kwargs):
        """
            This function starts a local ray cluster, splits data into equal sized
            subsets and trains one client/worker on each subset.
            Each client creates its own local density tree.
        """
        ray.init()
        train_jobs = []
        assign_jobs = []
        nodes = []
        self.args = args
        for c in range(args.num_clients):
            logging.info(f'Train node {c}')
            node = DensityTreeNode.remote(args.dataset)
            nodes.append(node)
            train_subset = train_data[c]
            subspace = feature_spaces[c]
            fts = feature_types[c]
            assign_jobs.append(node.assign_subset.remote(train_subset))
            assign_jobs.append(node.assign_feature_spaces.remote(subspace))
            assign_jobs.append(node.assign_feature_types.remote(fts))
        ray.get(assign_jobs)
        
        rtpt.step()

        for c in range(args.num_clients):
            train_jobs.append(nodes[c].train.remote())
        
        ray.get(train_jobs)
        rtpt.step()

        return nodes
    

    def build_fc_horizontal(self, nodes):
        """
            Collect all SPNs residing on clients and introduce a new
            root (sum node), weighted by dataset size on each client
        """
        # leafs = [SPNLeaf(c) for c in range(config.num_clients)]
        leaf_dict = [ray.get(node.get_models.remote()) for node in nodes]
        leafs = [list(l.values())[0][0] for l in leaf_dict]
        ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
        norm = sum(ds_len)
        weights = [d / norm for d in ds_len]
        leafs = [DensityLeaf(l) for l in leafs] # wrap density tree in SPN leaf
        spn = Sum(weights, leafs) 
        spn.scope = []
        for c in spn.children:
            spn.scope = list(set(spn.scope).union(set(c.scope)))
        spn = utils.reassign_node_ids(spn)
        return spn

    def build_fc_verhyb_naive(self, feature_subspaces, nodes):
        """
            Naively glue together client SPNs in vertical and hybrid setting.
            Each client holds one SPN.
            In hybrid setting, first the client SPNs which share the same feature
            space (scope) are connected by a mixture node.

            Then same as vertical case in which one Prodcut node is introduced
            which connects all client SPNs into one SPN.
        """
        spn = Product()
        added_nodes = [spn]
        for clients, subspace in feature_subspaces.items():
            if len(clients) > 1:
                s = Sum()
                leafs = []
                for c in clients:
                    # this yields an array with exactly one SPN included
                    leafs.append(ray.get(nodes[c].get_model.remote(tuple(subspace)))[0])
                # TODO: need custom leaf type(?)
                leaf_nodes = [DensityLeaf(leaf, subspace) for leaf in leafs]
                s.children = leaf_nodes
                s.weights = np.repeat(1/len(s.children), len(s.children))
                s.scope = set().union(*[set(l.scope) for l in leafs])
                spn.children += [s]
                spn.scope = set().union(*[c.scope for c in spn.children])
                added_nodes.append(s)
            else:
                node_idx = list(clients)[0]
                client_spn = ray.get(nodes[node_idx].get_spn.remote(tuple(subspace)))[0]
                spn.children += [client_spn]
                spn.scope = set().union(spn.scope, client_spn.scope)
        spn = utils.reassign_node_ids(spn)
        return spn, added_nodes
    
    def build_fc_verhyb_combinatorial(self, feature_subspaces, nodes):
        """
            Glue together client SPNs in vertical and hybrid setting.
            Each client holds N SPNs, each corresponding to one cluster.
            In hybrid setting, first the client SPNs which share the same feature
            space (scope) are put together in N mixtures, resulting in N new SPNs.

            Then same as vertical case in which one Prodcut node is introduced
            for each combinaion of clusters, followed by a mixture (root node)
            weighting the "combinatorial clusters".
        """
        leafs = []
        added_nodes = []
        for clients, subspace in feature_subspaces.items():
            if len(clients) > 1:
                # in hybrid case multple clients can hold same subspace
                client_models = []
                for c in clients:
                 # build mixture over common subspaces
                    models = ray.get(nodes[c].get_model.remote(tuple(subspace)))
                    client_models.append(models)
                
                num_spns = len(client_models[0]) # number is same for all clients
                joint_leafs = []
                for i in range(num_spns):
                    models = [cs[i] for cs in client_models]
                    sum = Sum()
                    leafs = [DensityLeaf(s, subspace) for s in models]
                    sum.children = leafs
                    sum.weights = np.repeat(1 / len(leafs), len(leafs))
                    scopes = []
                    for s in leafs:
                        scopes += list(s.scope)
                    sum.scope = list(set(scopes))
                    sum = utils.reassign_node_ids(sum)
                    joint_leafs.append(sum)
                added_nodes += joint_leafs
                leafs.append(joint_leafs)

            else:
                node_idx = list(clients)[0]
                client_spns = ray.get(nodes[node_idx].get_model.remote(tuple(subspace)))
                leafs.append(client_spns)
        
        fedspn = utils.build_fedspn_head(leafs)
        added_nodes = [fedspn] + added_nodes
        return fedspn, added_nodes

    def build_fc(self, feature_subspaces, nodes, args):
        """
            Construct FedSPN based on setting (horizontal, vertical or hybrid)
        """
        if len(feature_subspaces) == 1:
            # horizontal case
            spn = self.build_fc_horizontal(nodes)
            return spn, None
        else:
            # hybrid & vertical case
            if args.glueing == 'naive':
                spn, added_nodes = self.build_fc_verhyb_naive(feature_subspaces, nodes)
            elif args.glueing == 'combinatorial':
                spn, added_nodes = self.build_fc_verhyb_combinatorial(feature_subspaces, nodes)
            return spn, added_nodes

    def classify(self, model, test_data):
        labels = test_data[:, -1]
        tmp_lls = []
        u_labels = np.unique(labels)
        for l in u_labels:
            test_data[:, -1] = l
            lls = log_likelihood(model, test_data).flatten()
            tmp_lls.append(lls)
        tmp_lls = np.column_stack(tmp_lls)
        argmax = np.argmax(tmp_lls, axis=1).flatten()
        pred = np.array([u_labels[m] for m in argmax])
        return pred

# TODO: add NF implementation

class EinsumServer:

    def train(self, train_data, feature_spaces, args):
        """
            Train FedSPN in horizontal scenario with Einsum networks as SPN implementation on client side.
        """
        ray.init()
        train_jobs = []
        assign_jobs = []
        nodes = []
        for c in range(args.num_clients):
            logging.info(f'Train node {c}')
            node = EinetNode.remote(args.dataset, num_classes=10)
            nodes.append(node)
            if args.setting == 'horizontal':
                train_subset = train_data[c]
                subspace = feature_spaces[c]
            elif args.setting == 'vertical':
                subspace = feature_spaces[c]
                train_subset = train_data[c]
            elif args.setting == 'hybrid':
                subspace = feature_spaces[c]
                train_subset = train_data[c]
            assign_jobs.append(node.assign_subset.remote((subspace, train_subset)))
        ray.get(assign_jobs)
            
        rtpt.step()

        for c in range(args.num_clients):
            train_jobs.append(nodes[c].train.remote())
        
        ray.get(train_jobs)
        rtpt.step()

        return nodes

    def build_spn(self, feature_subspaces, nodes):
        """
            Build network-aligned SPN
        """
        if len(feature_subspaces) == 1:
            # horizontal case
            spn, client_spns = self.build_spn_horizontal(nodes)
        else:
            # hybrid & vertical case
            spn = Product()
            client_spns = []
            for clients, subspace in feature_subspaces.items():
                if len(clients) > 0:
                    s = Sum()
                    leafs = []
                    for c in clients:
                        client_spn = ray.get(nodes[c].get_spn.remote(subspace))
                        client_spns.append(client_spn)
                        leafs.append(SPNLeaf(subspace))
                    s.children = leafs
                    spn.children += [s]
                else:
                    node_idx = list(clients)[0]
                    client_spn = ray.get(nodes[node_idx].get_spn.remote(subspace))
                    client_spns.append(client_spn)
                    spn.children += [SPNLeaf(subspace)]

        return spn, client_spns
    
    def build_spn_horizontal(self, nodes):
        client_spns = [ray.get(node.get_spns.remote()) for node in nodes]
        client_spns = [list(d.values())[0] for d in client_spns]
        num_classes = client_spns[0].config.num_classes
        ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
        norm = sum(ds_len)
        weights = []
        for l in ds_len:
            w = [l / norm] * num_classes
            weights += w
        weights = torch.tensor([weights] * num_classes).T
        weights = weights.unsqueeze(0)
        weights = weights.unsqueeze(3)
        spn = SumLayer(num_classes*len(nodes), 1, num_classes)
        spn.weights = nn.Parameter(weights)
        return spn, client_spns

    def classify(self, client_spns, spn, test_data):
        """
            Classify test_data instances
        """
        accs, f1_micros, f1_macros = [], [], []
        for x, y in test_data:
            client_outs = torch.concat([s(x) for s in client_spns], dim=1)
            client_outs = client_outs.unsqueeze(1)
            client_outs = client_outs.unsqueeze(3)
            final_out = spn(client_outs).squeeze()
            pred = torch.argmax(final_out, dim=1)
            pred = pred.detach().numpy().flatten()
            acc = accuracy_score(y.numpy().flatten(), pred)
            f1_micro = f1_score(y.numpy().flatten(), pred, average='micro')
            f1_macro = f1_score(y.numpy().flatten(), pred, average='macro')
            accs.append(acc)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)

        return np.mean(accs), np.mean(f1_micros), np.mean(f1_macros)
    
def main_ddt(args):
    server = DenistyTreeServer()

    if args.setting == 'horizontal':
        train_data = get_horizontal_train_data(args.dataset, args.num_clients, 
                                               args.sample_partitioning, args.dir_alpha)
        feature_spaces = [[list(range(train_data[0].shape[1]))] for _ in range(args.num_clients)]
        feature_types = [[[ctxt.feature_types[args.dataset][f] for f in feature_spaces[0][0]]] for _ in range(args.num_clients)]
        nodes = server.train(train_data, feature_spaces, feature_types, args)
    elif args.setting == 'vertical':
        train_data, feature_spaces, labels = get_vertical_train_data(args.dataset, args.num_clients, return_labels=True)
        train_data = [np.hstack([td, labels.reshape(-1, 1)]) for td in train_data]
        feature_types = []
        for fs in feature_spaces:
            fts = [ctxt.feature_types[args.dataset][f] for f in fs]
            feature_types.append(fts)
        nodes = server.train(train_data, feature_spaces, feature_types, args)
    elif args.setting == 'hybrid':
        sample_frac = None if args.sample_frac == -1 else args.sample_frac
        train_data, feature_spaces, client_idx, labels = get_hybrid_train_data(args.dataset, args.num_clients, args.overlap_frac_hybrid, sample_frac, return_labels=True)
        final_train_data = []
        feature_types = []
        for i, (cidx, _) in enumerate(client_idx):
            td = train_data[i]
            client_labels = labels[cidx]
            final_train_data.append(np.hstack([td, client_labels.reshape(-1, 1)]))
            fs = feature_spaces[i]
            client_fts = []
            for s in fs:
                ft = [ctxt.feature_types[args.dataset][f] for f in s]
                client_fts.append(ft)
            feature_types.append(client_fts)
        nodes = server.train(final_train_data, feature_spaces, feature_types, args)

    # build fc
    grouped_feature_spaces = utils.group_clients_by_subspace(feature_spaces)
    model, added_nodes = server.build_fc(grouped_feature_spaces, nodes, args)

    # get accuracy
    test_data = get_test_data(args.dataset)
    # compute log-likelihood
    ll = np.mean(log_likelihood(model, test_data))
    # set last label column to nan for MPE
    labels = np.copy(test_data[:, -1])
    pred = server.classify(model, test_data)
    print(labels)
    print(pred)
    print(np.unique(pred))
    acc = accuracy_score(labels.flatten(), pred)
    f1_micro = f1_score(labels.flatten(), pred, average='micro')
    f1_macro = f1_score(labels.flatten(), pred, average='macro')
    rtpt.step()

    # shut down ray cluster
    ray.shutdown()

    return acc, f1_micro, f1_macro, ll


def main_rf(args):
    server = RandomForestServer()

    if args.setting == 'horizontal':
        train_data = get_horizontal_train_data(args.dataset, args.num_clients, 
                                               args.sample_partitioning, args.dir_alpha)
        feature_spaces = [list(range(train_data[0].shape[1] - 1)) for _ in range(args.num_clients)]
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'vertical':
        train_data, feature_spaces, labels = get_vertical_train_data(args.dataset, args.num_clients, return_labels=True)
        train_data = [np.hstack([td, labels.reshape(-1, 1)]) for td in train_data]
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'hybrid':
        sample_frac = None if args.sample_frac == -1 else args.sample_frac
        train_data, feature_spaces, client_idx, labels = get_hybrid_train_data(args.dataset, args.num_clients, args.overlap_frac_hybrid, sample_frac, return_labels=True)
        final_train_data = []
        for i, (cidx, _) in enumerate(client_idx):
            td = train_data[i]
            client_labels = labels[cidx]
            final_train_data.append(np.hstack([td, client_labels.reshape(-1, 1)]))
        nodes = server.train(final_train_data, feature_spaces, args)

    # build fc
    server.build_fc(nodes)

    # get accuracy
    test_data = get_test_data(args.dataset)
    labels = test_data[:, -1]
    pred = server.classify(test_data[:, :-1])
    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(labels.flatten(), pred)
    f1_micro = f1_score(labels.flatten(), pred, average='micro')
    f1_macro = f1_score(labels.flatten(), pred, average='macro')
    rtpt.step()

    # shut down ray cluster
    ray.shutdown()

    return acc, f1_micro, f1_macro, np.nan

def main_spflow(args):
    server = SPFlowServer()
    
    if args.sample_partitioning == 'iid':
        args.dir_alpha = 0.
    # train and create network aligned SPN
    if args.setting == 'horizontal':
        train_data = get_horizontal_train_data(args.dataset, args.num_clients, 
                                               args.sample_partitioning, args.dir_alpha)
        test_data = get_test_data(args.dataset)
        feature_spaces = [list(range(train_data[0].shape[1])) for _ in range(args.num_clients)]
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'vertical':
        train_data, feature_spaces = get_vertical_train_data(args.dataset, args.num_clients)
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'hybrid':
        sample_frac = None if args.sample_frac == -1 else args.sample_frac
        train_data, feature_spaces, client_idx = get_hybrid_train_data(args.dataset, args.num_clients, args.overlap_frac_hybrid, sample_frac)
        nodes = server.train(train_data, feature_spaces, args)

    grouped_feature_spaces = utils.group_clients_by_subspace(feature_spaces)
    na_spn, added_nodes = server.build_spn(grouped_feature_spaces, nodes, args)

    if args.setting in ['vertical', 'hybrid']:
        allowed_nodes_for_update = [n.id for n in added_nodes]
        if args.setting == 'hybrid':
            # align data based on client_idx
            idx_maps = []
            for cidx, cidx_view in client_idx:
                m = {c: cv for c, cv in zip(cidx, cidx_view)}
                idx_maps.append(m)
            samples = [set(list(c)) for c, _ in client_idx]
            intersection = list(set.intersection(*samples))
            cidx_client_views = []
            for idx in intersection:
                cidx_views = [m[idx] for m in idx_maps]
                cidx_client_views.append(cidx_views)
            cidx_client_views = np.array(cidx_client_views).T
            data = [td[idx] for td, idx in zip(train_data, cidx_client_views)]
            train_data = np.column_stack(data)
        else:
            train_data = np.column_stack(train_data)
        add_node_em_update(Sum, cond_sum_em_update(allowed_nodes_for_update))

        if args.algo == '2step':
            EM_optimization_network(na_spn, train_data)
        
        # simulate FedEM training
        elif args.algo == 'e2e':
            for node in get_nodes_by_type(na_spn, Sum):
                w_shape = len(node.weights)
                node.weights = np.random.uniform(0, 1, w_shape)
                node.weights = softmax(node.weights)

            EM_optimization(na_spn, train_data)


    # get accuracy
    test_data = get_test_data(args.dataset)
    # compute log-likelihood
    ll = np.mean(log_likelihood(na_spn, test_data))
    # set last label column to nan for MPE
    labels = np.copy(test_data[:, -1])
    pred = server.classify(na_spn, test_data)
    acc = accuracy_score(labels.flatten(), pred)
    f1_micro = f1_score(labels.flatten(), pred, average='micro')
    f1_macro = f1_score(labels.flatten(), pred, average='macro')
    rtpt.step()

    # shut down ray cluster
    ray.shutdown()

    return acc, f1_micro, f1_macro, ll

def main_einsum(args):
    """
        use RAT SPN client 
    """
    server = EinsumServer()

    if args.sample_partitioning == 'iid':
        args.dir_alpha = 0.
    # train and create network aligned SPN
    if args.setting == 'horizontal':
        train_data = get_horizontal_train_data(args.dataset, args.num_clients, 
                                               args.sample_partitioning, args.dir_alpha)
        feature_spaces = [list(range(train_data[0].shape[1])) for _ in range(args.num_clients)]
        train_data = make_data_loader(train_data, args.batch_size)
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'vertical':
        train_data, feature_spaces = get_vertical_train_data(args.dataset, args.num_clients)
        train_data = make_data_loader(train_data, args.batch_size)
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'hybrid':
        sample_frac = None if args.sample_frac == -1 else args.sample_frac
        train_data, feature_spaces = get_hybrid_train_data(args.ds, args.num_clients, args.min_dim_frac, args.max_dim_frac, sample_frac)
        train_data = make_data_loader(train_data, args.batch_size)
        nodes = server.train(train_data, feature_spaces, args)

    grouped_feature_spaces = utils.group_clients_by_subspace(feature_spaces)
    na_spn, client_spns = server.build_spn(grouped_feature_spaces, nodes)

    # get accuracy
    test_data = get_test_data(args.dataset)
    test_data = make_data_loader(test_data, args.batch_size)
    acc, f1_micro, f1_macro = server.classify(client_spns, na_spn, test_data)

    rtpt.step()

    # shut down ray cluster
    ray.shutdown()

    return acc, f1_micro, f1_macro

def main(args):
    # some sanity checks
    if args.cluster_by_label == 1:
        assert args.setting == 'horizontal', 'Cannot cluster by label in vertical and hybrid setting'

    table_dict = {'architecture': [], 'dataset': [], 'setting': [], 
                  'clients': [], 'accuracy': [], 'f1_micro': [], 'f1_macro': [], 
                  'skew': [], 'dir_alpha': [], 'll': []}

    if os.path.isfile('./experiments.csv'):
        df = pd.read_csv('./experiments.csv', index_col=0)
        table_dict = df.to_dict()
        table_dict = {k: list(v.values()) for k, v in table_dict.items()}
    for e in range(args.num_experiments):
        if args.model == 'spflow':
            acc, f1_micro, f1_macro, ll = main_spflow(args)
        elif args.model == 'rf':
            acc, f1_micro, f1_macro, ll = main_rf(args)
        #elif args.model == 'einsum':
        #    acc, f1_micro, f1_macro = main_einsum(args)
        elif args.model == 'ddt':
            acc, f1_micro, f1_macro, ll = main_ddt(args)
        else:
            raise ValueError("Implementation must be 'spflow' or 'einsum'")
        table_dict['architecture'].append(args.structure)
        table_dict['accuracy'].append(acc)
        table_dict['clients'].append(args.num_clients)
        table_dict['dataset'].append(args.dataset)
        table_dict['f1_macro'].append(f1_macro)
        table_dict['f1_micro'].append(f1_micro)
        table_dict['setting'].append(args.setting)
        table_dict['skew'].append(args.sample_partitioning)
        table_dict['dir_alpha'].append(args.dir_alpha)
        table_dict['ll'].append(ll)
    df = pd.DataFrame.from_dict(table_dict)
    df.to_csv('./experiments.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='horizontal')
parser.add_argument('--algo', default='2step')
parser.add_argument('--num-clients', type=int, default=5)
parser.add_argument('--dataset', default='income')
parser.add_argument('--task', default='classification')
parser.add_argument('--sample-partitioning', default='iid')
parser.add_argument('--structure', default='learned')
parser.add_argument('--overlap-frac-hybrid', default=0.3, type=float)
parser.add_argument('--sample-frac', default=-1., type=float)
parser.add_argument('--num-experiments', default=1, type=int)
parser.add_argument('--dir-alpha', default=0.0, type=float)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-clusters', default=2, type=int)
parser.add_argument('--glueing', default='combinatorial')
parser.add_argument('--model', default='spflow')
parser.add_argument('--cluster-by-label', default=0, type=int)

args = parser.parse_args()

main(args)