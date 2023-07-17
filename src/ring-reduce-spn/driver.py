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
import torch
from client import Node
from datasets import get_medical_data
import networkx as nx
from typing import List

def train():
    ray.init()
    run_id = str(round(time.time() * 1000))
    chk_dir = f'./checkpoints/{run_id}'
    train_data, test_data = get_medical_data()
    train_data_idx = np.arange(len(train_data))
    test_data_idx = np.arange(len(test_data))
    nodes = []

    for c in range(10):
        node = Node.remote('medical', chk_dir + f'/client_{c}', 3, c)
        train_subset = np.random.choice(train_data_idx, int(len(train_data) / 10), False)
        test_subset = np.random.choice(test_data_idx, int(len(test_data) / 10), False)
        train_data_idx = np.array([x for x in train_data_idx if x not in train_subset])
        test_data_idx = np.array([x for x in test_data_idx if x not in test_subset])    
        node.assign_subset.remote(list(train_subset), list(test_subset))
        nodes.append(node)

    einets = [n.train.remote() for n in nodes]
    return nodes, einets

def infer(virtual_graph, query, nodes):
    """
        This function traverses along the virtual graph which resembles
        a SPN-like structure defined over the communication network.
        The query is passed to each client for evaluation.
    """
    # sort topologically
    nodes = nx.topological_sort(virtual_graph)
    n = nodes[0]
    if query['type'] == 'joint':
        res = evaluate_dfs_recursive(virtual_graph, n, query['x'], nodes)
    elif query['type'] == 'cond':
        # P(A | B) = P(A, B) / P(B)
        joint = evaluate_dfs_recursive(virtual_graph, n, query['x'], nodes)
        marginal = evaluate_dfs_recursive(virtual_graph, n, query['c'], nodes)
        res = joint / marginal
    return res

def evaluate_dfs_recursive(G: nx.DiGraph, n, rv_vals, nodes: List[Node]):
    """
        Recursively traverse virtual graph of communication network-aligned SPN
        in depthf-first search manner.
    """
    if len(list(G.successors(n))) == 0:
        result = nodes[n].query.remote(rv_vals)
        return result
    
    results = []
    for s in G.successors(n):
        r = evaluate_dfs_recursive(G, s, rv_vals, nodes)
        results.append(r)
    
    if n.reduce == 'sum':
        # exp->sum->log
        w = torch.tensor(n.weights)
        probs = torch.exp(torch.tensor(results))
        ws = w @ probs.T
        return torch.log(ws)
    elif n.reduce == 'prod':
        # we get log-likelihoods from each leaf -> sum
        return torch.sum(torch.tensor(results))