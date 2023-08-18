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

def region_graph_to_spn(G):
    """
    Convert a region graph into an SPN
    """
    nodes = reversed(list(nx.topological_sort(G)))
    

def random_region_graph(depth, variables, curr_parents, G=nx.DiGraph()):
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
        idx = [i]
        for j, s in enumerate(subspaces):
            if space == s:
                idx.append(j)
        client_dict[tuple(idx)] = space
    return client_dict
