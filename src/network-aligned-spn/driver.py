"""
    This file starts the driver process of a ray cluster.
    It then coordinates training of the clients. 
    Note that this just simulates the ring-reduce network topology
    proposed in our FedSPN paper.
    Nevertheless, semantically it's performing the same operations
    as ring reduce algorithm.
"""

import ray
import time
import numpy as np
from client import FlowNode
from datasets.datasets import Avazu, Income
from spn_leaf import SPNLeaf
from spn.structure.Base import Sum, Product
from spn.algorithms.Inference import log_likelihood, sum_log_likelihood
from rtpt import RTPT
import config
import logging
import sys
import warnings
import argparse

warnings.filterwarnings('ignore')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

rtpt = RTPT('JS', 'FedSPN Driver', 3)
rtpt.start()

def train(train_data):
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

    for c in range(config.num_clients):
        logging.info(f'Train node {c}')
        node = FlowNode.remote('income', 1, rank=1)
        nodes.append(node)
        train_subset = np.random.choice(train_data_idx, int(len(train_data) / config.num_clients), False)
        train_data_idx = np.array([x for x in train_data_idx if x not in train_subset])  
        assign_jobs.append(node.assign_subset.remote(list(train_subset)))
    ray.get(assign_jobs)
    rtpt.step()

    for c in range(config.num_clients):
        train_jobs.append(nodes[c].train.remote(return_spn=False))
    
    ray.get(train_jobs)
    rtpt.step()

    return nodes

def infer(spn, query, nodes):
    """
    evaluate network-aligned SPN
    """
    node_res_ref = [n.query.remote(query) for n in nodes]
    node_res = [ray.get(nrr) for nrr in node_res_ref]
    network_aligned_in_data = np.array(node_res).reshape(1, -1)
    return log_likelihood(spn, network_aligned_in_data)

def build_spn_horizontal(nodes):
    # TODO: Build RAT-SPN
    leafs = [SPNLeaf(c) for c in range(config.num_clients)]
    ds_len = [ray.get(node.get_dataset_len.remote()) for node in nodes]
    norm = sum(ds_len)
    weights = [d / n for d, n in zip(ds_len, norm)]
    spn = Sum(weights, leafs)
    return spn

def build_spn_hybrid(feature_subspaces):
    spn = Product()
    for clients, intersct in feature_subspaces:
        if len(clients) > 0:
            s = Sum()
            subspns = [SPNLeaf(intersct) for _ in clients]
            s.children = subspns
            spn.children += [s]
        else:
            client_spn = SPNLeaf(intersct)
            spn.children += [client_spn]

def main(args):

    # load data 
    train_data = Income('../../datasets/income/', split='train').features

    # train SPNs
    nodes = train(train_data)

    # create network-aligned SPN
    if args.setting == 'horizontal':
        na_spn = build_spn_horizontal()
    elif args.setting == 'hybrid':
        # TODO: get feature sub spaces
        na_spn = build_spn_hybrid()
    elif args.setting == 'vertical':
        # TODO: implement
        pass

    # get probability of some test sample
    test_data = Income('../../datasets/income/', split='test')
    sample_idx = np.random.randint(0, len(test_data))
    x_test = test_data[sample_idx].numpy()
    query_dict = ray.put({i: v for i, v in enumerate(x_test)})
    p = infer(na_spn, query_dict, nodes)
    rtpt.step()

    print(p)


parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='horizontal')
parser.add_argument('--num-clients', default=5)
parser.add_argument('--dataset', default='income')

args = parser.parse_args()

main(args)