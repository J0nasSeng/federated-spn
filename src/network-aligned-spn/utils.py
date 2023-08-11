import numpy as np
import os
import torch
import errno
from PIL import Image
from numproto import proto_to_ndarray
from torch.utils.data import DataLoader
import networkx as nx
import itertools

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

def build_rand_region_graph(r, variables):
    """
        Build a region graph s.t. each region has one child (partition) and
        each partition has two childs (regions).

        TODO: Isn't a binary tree too simple? shouldn't there be interactions?

        :param r: number of region layers
        :param d: number of partition childs being created for each region
    """
    G = nx.DiGraph()
    root = Region(variables)
    G.add_node(root)
    curr_children = [root]
    for depth in range(1, r):
        new_children = []
        for c in curr_children:
            if not len(c.scope) == 1:
                split_idx = np.random.randint(1, len(c.scope) - 1)
                s1, s2 = c.scope[:split_idx], c.scope[split_idx:]
                p = Partition(c.scope)
                r1 = Region(s1)
                r2 = Region(s2)
                G.add_edge(c, p)
                G.add_edge(p, r1)
                G.add_edge(p, r2)
                new_children += [r1, r2]
        curr_children = new_children
    return G

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