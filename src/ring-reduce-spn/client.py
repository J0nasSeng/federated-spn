"""
    This file contains the code for FedSPN clients.
    It is implemented as a ray actor.
"""
import ray
from ..einsum.EinsumNetwork import EinsumNetwork, Args
from ..einsum.Graph import poon_domingos_structure
from torch.utils.data import Subset
import config
import json

@ray.remote
class Node:

    def __init__(self, dataset, inds_path, rank=None) -> None:
        self.rank = rank
        self.dataset = dataset
        self.inds_path = inds_path
        self.einet = init_spn(f'cuda:{rank}')

    def query(self, query):
        """
            Query SPN
        """

    def train(self):
        """
            Train SPN on local data
        """

    def _load_data(self):
        """
            Load data
        """



def init_spn(device):
    """
        Build a SPN (implemented as an einsum network). The structure is either
        the same as proposed in https://arxiv.org/pdf/1202.3732.pdf (referred to as
        poon-domingos) or a binary tree.

        In case of poon-domingos the image is split into smaller hypercubes (i.e. a set of
        neighbored pixels) where each pixel is a random variable. These hypercubes are split further
        until we operate on pixel-level. The spplitting is done randomly. For more information
        refer to the link above.
    """

    if config.structure == 'poon-domingos':
        pd_delta = [[config.height / d, config.width / d] for d in config.pd_num_pieces]
        graph = poon_domingos_structure(shape=(config.height, config.width), delta=[4], axes=[pd_delta])
    else:
        raise AssertionError("Unknown Structure")

    args = Args(
            num_var=config.num_vars,
            num_dims=config.num_dims,
            num_classes=1,
            num_sums=config.K,
            num_input_distributions=config.K,
            exponential_family=config.exponential_family,
            exponential_family_args=config.exponential_family_args,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)

    einet = EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    return einet

def make_subset(dataset, split, rank, inds_file):
    with open(inds_file) as f:
        inds_dict = json.load(f)
    indices = inds_dict[rank][split]
    subset = Subset(dataset, indices)
    return subset    