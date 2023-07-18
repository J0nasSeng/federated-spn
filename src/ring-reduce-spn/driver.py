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
from client import Node
from datasets import get_medical_data
from spn_leaf import SPNLeaf
from spn.structure.Base import Sum, Product
from spn.algorithms.Inference import log_likelihood
import config
import logging
import sys
import os

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

def train(train_data, test_data):
    """
        This function starts a local ray cluster, splits data into equal sized
        subsets and trains one client/worker on each subset.
        Each client creates its own local SPN which can then be queried using
        node.query(...)
    """
    ray.init()
    run_id = str(round(time.time() * 1000))
    chk_dir = f'./checkpoints/{run_id}'
    train_data_idx = np.arange(len(train_data))
    test_data_idx = np.arange(len(test_data))
    train_jobs = []
    assign_jobs = []
    nodes = []

    for c in range(config.num_clients):
        logging.info(f'Train node {c}')
        node = Node.remote('medical', chk_dir + f'/client_{c}', 1, num_epochs=3, rank=1)
        nodes.append(node)
        train_subset = np.random.choice(train_data_idx, int(len(train_data) / config.num_clients), False)
        test_subset = np.random.choice(test_data_idx, int(len(test_data) / config.num_clients), False)
        train_data_idx = np.array([x for x in train_data_idx if x not in train_subset])
        test_data_idx = np.array([x for x in test_data_idx if x not in test_subset])    
        assign_jobs.append(node.assign_subset.remote(list(train_subset), list(test_subset)))
    ray.wait(assign_jobs, num_returns=config.num_clients)

    for c in range(config.num_clients):
        train_jobs.append(nodes[c].train.remote(return_spn=False)) 

    print(train_jobs)
    ray.wait(train_jobs, num_returns=config.num_clients)
    for n in nodes:
        print(ray.get(n.get_losses.remote()))

    return nodes

def infer(spn, query, nodes):
    """
    evaluate network-aligned SPN
    """
    node_res_ref = [n.query.remote(query) for n in nodes]
    node_res = [ray.get(nrr) for nrr in node_res_ref]
    network_aligned_in_data = np.array(node_res).reshape(1, -1)
    return log_likelihood(spn, network_aligned_in_data)


def build_network_aligned_spn():
    leafs = [SPNLeaf(c) for c in range(config.num_clients)]
    weights = [1/config.num_clients]*config.num_clients
    spn = Sum(weights, leafs)
    return spn

# load data 
train_data, test_data = get_medical_data()

# train SPNs
nodes = train(train_data, test_data)

# create network-aligned SPN
na_spn = build_network_aligned_spn()

# get probability of some test sample
sample_idx = np.random.randint(0, len(test_data))
x_test, _ = test_data[sample_idx]
query_dict = ray.put({i: v for i, v in enumerate(x_test)})
p = infer(na_spn, query_dict, nodes)

print(p)