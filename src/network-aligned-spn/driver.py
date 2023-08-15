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
from datasets.utils import get_data
from spn_leaf import SPNLeaf
from spn.structure.Base import Sum, Product, get_nodes_by_type
from spn.algorithms.Inference import log_likelihood, sum_log_likelihood
from spn.algorithms.MPE import mpe
from rtpt import RTPT
import config
import logging
import sys
import warnings
import argparse
from sklearn.metrics import accuracy_score
import utils

warnings.filterwarnings('ignore')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

rtpt = RTPT('JS', 'FedSPN Driver', 3)
rtpt.start()

def train_horizontal(train_data, args):
    """
        This function starts a local ray cluster, splits data into equal sized
        subsets and trains one client/worker on each subset.
        Each client creates its own local SPN which can then be queried using
        node.query(...)
    """
    ray.init()
    train_data_idx = np.arange(len(train_data))
    train_jobs = []
    assign_jobs = []
    nodes = []

    for c in range(args.num_clients):
        logging.info(f'Train node {c}')
        node = FlowNode.remote(args.dataset)
        nodes.append(node)
        train_subset = np.random.choice(train_data_idx, int(len(train_data) / args.num_clients), False)
        train_data_idx = np.array([x for x in train_data_idx if x not in train_subset])  
        assign_jobs.append(node.assign_subset.remote(list(train_subset)))
    ray.get(assign_jobs)

    feature_space = list(range(0, train_data.shape[1]))
    assign_jobs = []
    for c in range(args.num_clients):
        if args.setting == 'horizontal':
            subspace = feature_space
        node = nodes[c]
        assign_jobs.append(node.assign_subspace.remote(subspace))
    ray.get(assign_jobs)
        
    rtpt.step()

    for c in range(args.num_clients):
        train_jobs.append(nodes[c].train.remote())
    
    ray.get(train_jobs)
    rtpt.step()

    return nodes

def ll(spn, query, nodes):
    """
    evaluate network-aligned SPN
    """
    node_res_ref = [n.query.remote(query) for n in nodes]
    node_res = [ray.get(nrr) for nrr in node_res_ref]
    network_aligned_in_data = np.array(node_res).reshape(1, -1)
    return log_likelihood(spn, network_aligned_in_data)

def most_probable_value(spn, input):
    return mpe(spn, input)

def build_spn_horizontal(nodes):
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

def build_spn(feature_subspaces, nodes):
    if len(feature_subspaces) == 1:
        # horizontal case
        spn = build_spn_horizontal(nodes)
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

def main(args):

    # load data 
    train_data = get_data(args.dataset, 'train')
    
    # train and create network aligned SPN
    if args.setting == 'horizontal':
        feature_space = {tuple(list(range(args.num_clients))): list(range(train_data.shape[1]))}
        nodes = train_horizontal(train_data, args)
    elif args.setting == 'hybrid':
        # TODO: get feature sub spaces
        pass
    elif args.setting == 'vertical':
        # TODO: implement
        pass

    na_spn = build_spn(feature_space, nodes)

    # get accuracy    
    test_data = get_data(args.dataset, 'test')
    # set last label column to nan for MPE
    labels = np.copy(test_data[:, -1])
    test_data[:, -1] = np.nan
    pred = most_probable_value(na_spn, test_data)
    acc = accuracy_score(labels.flatten(), pred[:,-1].flatten())
    rtpt.step()

    print(acc)


parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='horizontal')
parser.add_argument('--num-clients', type=int, default=5)
parser.add_argument('--dataset', default='income')
parser.add_argument('--task', default='classification')

args = parser.parse_args()

main(args)