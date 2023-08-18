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
from client import FlowNode
from datasets.utils import get_horizontal_train_data, get_test_data, get_vertical_train_data, get_hybrid_train_data
from spn.structure.Base import Sum, Product
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from rtpt import RTPT
import logging
import sys
import warnings
import argparse
from sklearn.metrics import accuracy_score
import utils
from spn_leaf import SPNLeaf
from einet.layers import Sum as SumLayer
import torch

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
            node = FlowNode.remote(args.dataset)
            nodes.append(node)
            if args.setting == 'horizontal':
                train_subset = train_data[c]
                subspace = list(feature_spaces.values())[0]
            elif args.setting == 'vertical':
                subspace = feature_spaces[c]
                conc_train_data = np.concatenate(train_data)
                train_subset = conc_train_data[:, subspace]
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

    def classify(self, spn, test_data):
        test_data[:, -1] = np.nan
        pred = mpe(spn, test_data)
        return pred[:, -1].flatten()

    def build_spn_horizontal(self, nodes):
        # leafs = [SPNLeaf(c) for c in range(config.num_clients)]
        leaf_dict = [ray.get(node.get_spns.remote()) for node in nodes]
        leafs = [list(l.values())[0] for l in leaf_dict]
        ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
        norm = sum(ds_len)
        weights = [d / norm for d in ds_len]
        spn = Sum(weights, leafs)
        spn.scope = []
        for c in spn.children:
            spn.scope = list(set(spn.scope).union(set(c.scope)))
        spn = utils.reassign_node_ids(spn)
        return spn

    def build_spn(self, feature_subspaces, nodes):
        if len(feature_subspaces) == 1:
            # horizontal case
            spn = self.build_spn_horizontal(nodes)
        else:
            # hybrid & vertical case
            spn = Product()
            for clients, subspace in feature_subspaces.items():
                if len(clients) > 0:
                    s = Sum()
                    leafs = []
                    for c in clients:
                        leafs.append(ray.get(nodes[c].get_spn.remote(subspace)))
                    s.children = leafs
                    spn.children += [s]
                else:
                    node_idx = list(clients)[0]
                    client_spn = ray.get(nodes[node_idx].get_spn.remote(subspace))
                    spn.children += [client_spn]

        return spn

class EinsumServer:

    def train_horizontal(train_data, feature_spaces, args):
        """
            Train FedSPN in horizontal scenario with Einsum networks as SPN implementation on client side.
        """
        ray.init()
        train_jobs = []
        assign_jobs = []
        nodes = []

        for c in range(args.num_clients):
            logging.info(f'Train node {c}')
            node = FlowNode.remote(args.dataset)
            nodes.append(node)
            train_subset = train_data[c]
            assign_jobs.append(node.assign_subset.remote(train_subset))
        ray.get(assign_jobs)

        feature_space = list(feature_spaces.values())[0]
        assign_jobs = []
        for c in range(args.num_clients):
            node = nodes[c]
            assign_jobs.append(node.assign_subspace.remote(feature_space))
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
        ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
        norm = sum(ds_len)
        weights = [d / norm for d in ds_len]
        num_classes = client_spns[0].config.num_classes
        spn = SumLayer(num_classes*len(nodes), 1, num_classes)
        spn.weights = torch.tensor(weights)
        return spn, client_spns

    def classify(self, client_spns, spn, test_data):
        """
            Classify test_data instances
        """
        client_outs = torch.concat([s(test_data) for s in client_spns], dim=1)
        final_out = spn(client_outs)
        return torch.argmax(final_out, dim=1)


def main_learned_structure(args):
    server = SPFlowServer()
    
    # train and create network aligned SPN
    if args.setting == 'horizontal':
        train_data = get_horizontal_train_data(args.dataset, args.num_clients, args.sample_partitioning)
        feature_spaces = [list(range(train_data[0].shape[1])) for _ in range(args.num_clients)]
        nodes = server.train(train_data, feature_spaces, args)
    elif args.setting == 'hybrid':
        train_data, feature_spaces = get_vertical_train_data(args.dataset, args.num_clients)
        nodes = server.train(train_data, feature_spaces)
    elif args.setting == 'vertical':
        sample_frac = None if args.sample_frac == -1 else args.sample_frac
        train_data, feature_spaces = get_hybrid_train_data(args.ds, args.num_clients, args.min_dim_frac, args.max_dim_frac, sample_frac)
        nodes = server.train(train_data, feature_spaces)


    grouped_feature_spaces = utils.group_clients_by_subspace(feature_spaces)
    na_spn = server.build_spn(grouped_feature_spaces, nodes)

    # get accuracy
    test_data = get_test_data(args.dataset)
    # set last label column to nan for MPE
    labels = np.copy(test_data[:, -1])
    pred = server.classify(na_spn, test_data)
    acc = accuracy_score(labels.flatten(), pred)
    rtpt.step()

    print(acc)

def main_rat_structure(args):
    """
        use RAT SPN client 
    """


def main(args):

    if args.structure == 'learned':
        main_learned_structure(args)
    elif args.structure == 'rat':
        main_rat_structure(args)
    else:
        raise ValueError("structure must be either 'rat' or 'learned'")


parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='horizontal')
parser.add_argument('--num-clients', type=int, default=5)
parser.add_argument('--dataset', default='income')
parser.add_argument('--task', default='classification')
parser.add_argument('--sample-partitioning', default='iid')
parser.add_argument('--structure', default='learned')
parser.add_argument('--min-dim-frac', default=0.25, type=float)
parser.add_argument('--max-dim-frac', default=0.5, type=float)
parser.add_argument('--sample-frac', default=-1., type=float)


args = parser.parse_args()

main(args)