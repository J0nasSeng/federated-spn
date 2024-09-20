import numpy as np
import os
import torch
import errno
from PIL import Image
from numproto import proto_to_ndarray
from torch.utils.data import DataLoader
import networkx as nx
import itertools
from spn.structure.Base import get_nodes_by_type
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from itertools import product
from scipy.special import softmax

def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx
    

class Region:

    def __init__(self, scope):
        self.scope = scope

class Partition:

    def __init__(self, scope) -> None:
        self.scope = scope

def region_graph_to_spn(G: nx.DiGraph, curr_layer, scope_dist_mapping, spn_root=None, num_sums=5):
    next_layer = []
    spn = spn_root
    if len(curr_layer) == 0:
        return spn
    for i, node in enumerate(curr_layer):
        if len(list(G.pred[node])) == 0:
            # then we have root node
            spn = Sum()
            spn.weights = np.random.normal(0, 0.1, num_sums**2)
            spn.weights = softmax(spn.weights)
            spn.scope = node.scope
            node.spn_nodes = [spn]
            next_layer = list(G.succ[node])
        else:
            if type(node) == Partition:
                prods = [Product() for _ in range(num_sums**2)]
                for p in prods:
                    p.scope = node.scope
                pred = list(G.pred[node])[0]
                for s in pred.spn_nodes:
                    s.children = prods
                next_layer += list(G.succ[node])
                node.spn_nodes = prods
            elif type(node) == Region:
                succ = list(G.succ[node])
                if len(succ) == 0:
                    # leaf node
                    scope = node.scope[0]
                    Distribution, params = scope_dist_mapping[scope]
                    sums = [Distribution(**params) for _ in range(num_sums)]
                else:
                    sums = [Sum() for _ in range(num_sums)]
                for s in sums:
                    s.scope = node.scope
                    if type(s) == Sum:
                        s.weights = np.random.normal(0, 0.1, num_sums**2)
                        s.weights = softmax(s.weights)
                pred = list(G.pred[node])[0]
                # TODO: use pytorch implementation of SPN
                for k, j in product(list(range(num_sums)), list(range(num_sums))):
                    s = sums[k]
                    if i % 2 == 0:
                        # left side
                        idx = (k*num_sums)+j
                    else:
                        # right side
                        idx = (j*num_sums)+k
                    pred.spn_nodes[idx].children.append(s)
                node.spn_nodes = sums
                next_layer += succ
    return region_graph_to_spn(G, next_layer, scope_dist_mapping, spn, num_sums)
    

def random_region_graph(depth, variables, curr_parents, G=None):
    if G is None:
        G = nx.DiGraph()
    if depth == 0:
        root = Region(variables)
        G.add_node(root)
        parents = [root]
        return random_region_graph(depth+1, variables, parents, G)
    else:
        next_parents = []
        for p in curr_parents:
            if type(p) == Region:
                # add a partition node
                if len(p.scope) > 1:
                    node = Partition(p.scope)
                    G.add_edge(p, node)
                    next_parents.append(node)
            elif type(p) == Partition:
                # split scope and add two regions
                if len(p.scope) == 2:
                    choice_left = [p.scope[0]]
                    choice_right = [p.scope[1]]
                else:
                    choice_left = list(np.random.choice(p.scope, int(len(p.scope) / 2), False))
                    choice_right = [v for v in p.scope if v not in choice_left]
                node1, node2 = Region(choice_left), Region(choice_right)
                G.add_edge(p, node1)
                G.add_edge(p, node2)
                next_parents += [node1, node2]
        if len(next_parents) == 0:
            return G
        else:
            return random_region_graph(depth+1, variables, next_parents, G)


def split_feature_space(space, clients):
    """
        Find disjoint subspaces s.t. clients can train SPNs independently and 
        server can merge them in a pre-defined structure
    """
    p = list(itertools.combinations(clients))
    p.sort(key=len)
    feature_subsets = []
    for rho in p:
        client_features = [c.get_feature_space.remote() for c in rho]
        intersct = set.intersection(*map(set, client_features))
        if len(intersct) > 0:
            feature_subsets.append((rho, intersct))
    
    for rho, subspace in feature_subsets:
        for c in rho:
            c.add_subspace.remote(subspace)


def get_data_by_cluster(dataloader: DataLoader, clusters, idx, cluster_n):
    data_idx = np.argwhere(clusters == cluster_n).flatten()
    subset_idx = idx[data_idx]
    return subset_idx

def reassign_node_ids(spn):
    nodes = get_nodes_by_type(spn)
    for id, n in enumerate(nodes):
        n.id = id
    return spn
        
def group_clients_by_subspace(subspaces):
    client_dict = {}
    for i, space in enumerate(subspaces):
        idx = []
        for j, s in enumerate(subspaces):
            if space == s:
                idx.append(j)
        if len(idx) == 0:
            # no match was found
            idx = [i]
        client_dict[tuple(idx)] = space
    return client_dict

def adjust_scope(spn, space):
    nodes = get_nodes_by_type(spn)
    scope_mapping = {i: s for i, s in enumerate(space)}
    for n in nodes:
        sc = list(n.scope)
        new_sc = [scope_mapping[i] for i in sc]
        n.scope = new_sc
    return spn

def build_fedspn_head(client_cluster_spns):
    num_clients = len(client_cluster_spns)
    # assume num clusters is equal on all clients 
    num_clusters = len(client_cluster_spns[0])
    clusters = list(range(num_clusters))
    prods = {}
    for l in range(1, num_clients):
        for comb in product(*[clusters]*num_clients):
            prefix = list(comb)[:l]
            next_node = list(comb)[l]
            prod_id = tuple(prefix + [next_node])
            if l > 1:
                # connect product node of last layer with next_node's SPN of l-the client
                relevant_spns = [prods[tuple(prefix)], client_cluster_spns[l][next_node]]
            else:
                # first product layer -> connect all client SPNs of a certain prod_id
                relevant_spns = [client_cluster_spns[i][j] for i,j in enumerate(prod_id)]
            scopes = [set(s.scope) for s in relevant_spns]
            prod_scope = list(set().union(*scopes))
            prod = Product(relevant_spns)
            prod.scope = prod_scope
            prods[prod_id] = prod

    all_scopes = set()
    for cluster_spns in client_cluster_spns:
        for s in cluster_spns:
            all_scopes = all_scopes.union(set(s.scope))
    
    root_children = [n for prefix, n in prods.items() if len(prefix) == num_clients]
    weights = softmax(np.zeros(len(root_children)))
    #weights = softmax(np.random.normal(0, 0.5, len(root_children)))
    root = Sum(weights, root_children)
    root.scope = list(all_scopes)
    root = reassign_node_ids(root)
    return root

def infer_node_type(data, uniqe_limit=100):
    types = []
    for i  in range(data.shape[1]):
        unique = len(np.unique(data[:, i]))
        if unique < uniqe_limit:
            params = {'p': np.repeat(1 / unique, unique)}
            types.append((Categorical, params))
        else:
            params = {'mean': 0, 'stdev': 1}
            types.append((Gaussian, params))
    return types

def log_likelihoods(outputs, targets=None):
    """Compute the likelihood of an Einet."""
    if targets is None:
        num_roots = outputs.shape[-1]
        if num_roots == 1:
            lls = outputs
        else:
            num_roots = torch.tensor(float(num_roots), device=outputs.device)
            lls = torch.logsumexp(outputs - torch.log(num_roots), -1)
    else:
        lls = outputs.gather(-1, targets.unsqueeze(-1))
    return lls