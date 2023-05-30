import flwr as fl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from einsum import EinsumNetwork, Graph
import config
from datasets import get_dataset_loader
from rtpt import RTPT
from collections import OrderedDict
import networkx as nx
import argparse
import numpy as np
from utils import flwr_params_to_numpy, save_image_stack
import os
import matplotlib.pyplot as plt

def main(dataset, num_clients, client_id, device):

    data_loader = get_dataset_loader(dataset, num_clients, config.dataset_inds_file, config.data_skew)
    train_data, val_data = data_loader.load_client_data(client_id)
    rtpt = RTPT('JS', 'FedSPN-Client', 10)
    rtpt.start()

    class SPNClient(fl.client.NumPyClient):
        """
            This class implements a client which trains a SPN (locally).
            After training (one fit-call), it sends its learned parameters 
            to the server. 
        """

        def __init__(self):
            self.device = device
            self.train_loader = DataLoader(train_data, config.batch_size, True)
            self.val_loader = DataLoader(val_data, config.batch_size, True)
            self.einet = init_spn(device)
            for layer in self.einet.einet_layers:
                params = list(layer.parameters())[0]
                print(layer)
                print(params.shape)

        def set_parameters(self, parameters):
            """
                Set parameters of SPN.
            """

        def get_parameters(self):
            return spn_to_param_list(self.einet)

        def fit(self, parameters, cfg):
            """
                Fit SPN and send parameters to server
            """
            #self.einet = train(self.einet, self.train_loader, config.num_epochs, device)

            # collect parameters and send back to server
            params = self.get_parameters()
            return params, len(train_data), {}
        
        def evaluate(self, parameters, cfg):
            """
                Evaluate the local SPN or global SPN (depending on what's sent by server)
            """
            einet = make_spn(parameters)
            # TODO: what to evaluate here?
            # for now, just save model
            samples = einet.sample(num_samples=25).cpu().numpy()
            samples = samples.reshape((-1, 28, 28))
            save_image_stack(samples, 5, 5, os.path.join('../samples/demo_mnist/', "samples.png"), margin_gray_val=0.)
            torch.save(einet, './model.pt')

    # Start client
    fl.client.start_numpy_client(server_address="[::]:{}".format(config.port), client=SPNClient())

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
        graph = Graph.poon_domingos_structure(shape=(config.height, config.width), delta=pd_delta)
    elif config.structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=config.num_vars, depth=config.depth, num_repetitions=config.num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=config.num_vars,
            num_dims=1,
            num_classes=1,
            num_sums=config.K,
            num_input_distributions=config.K,
            exponential_family=config.exponential_family,
            exponential_family_args=config.exponential_family_args,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    return einet

def train(einet, train_loader, num_epochs, device):

    """
    Training loop to train the SPN. Follows EM-procedure.
    """

    for epoch_count in range(num_epochs):
        einet.train()

        total_ll = 0.0
        for x, y in train_loader:
            x = x.reshape(x.shape[0], config.num_vars)
            x = x.to(device)
            outputs = einet.forward(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()

        einet.em_update()
    return einet

def make_spn(params):
    """
        Given a parameter array, reconstruct the learnt SPN.
        
        params: List containing a description of the structure in form of
            a list of lists (=param[0]) and the SPN's parameters (=params[1])
    """
    adj, meta_info = params[-2], params[-1]
    parameters = params[:-2]

    # 0 = Product node, 1 = Leaf node, 2 = Sum node
    graph = nx.DiGraph()

    # reconstruct graph based on adjacency, node-types and scopes
    # passed by client
    for info in meta_info:
        node_type = info[0]
        info = info[1:]
        scope = list(info[info != -1])
        if node_type == 0:
            node = Graph.Product(scope)
        else:
            node = Graph.DistributionVector(scope)
        graph.add_node(node)

    nodes = list(graph.nodes)
    for row in range(adj.shape[0]):
        for col in range(adj.shape[0]):
            if adj[row, col] == 1:
                src, dst = nodes[row], nodes[col]
                graph.add_edge(src, dst)

    for node in Graph.get_leaves(graph):
        node.einet_address.replica_idx = 0
    
    args = EinsumNetwork.Args(
        num_var=config.num_vars,
        num_dims=1,
        num_classes=1,
        num_sums=config.K,
        num_input_distributions=config.K,
        exponential_family=config.exponential_family,
        exponential_family_args=config.exponential_family_args,
        online_em_frequency=config.online_em_frequency,
        online_em_stepsize=config.online_em_stepsize)

    # create einsum object and set parameters as sent by client
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    # first initialize (relveant to fill buffers)
    einet.initialize()
    # set all parameters requiring gradient
    with torch.no_grad():
        for eparam, param in zip(einet.parameters(), parameters):
            if eparam.requires_grad:
                eparam.copy_(torch.tensor(param))

    return einet

def spn_to_param_list(einet: EinsumNetwork.EinsumNetwork):
    """
        Transform a EinsumNetwork object to a parameter-array containing
        the structure as a list of lists and the parameters as a separate array.
    """

    adj = nx.convert_matrix.to_numpy_array(einet.graph)
    node_meta_info = []
    max_scope_len = 0
    for node in einet.graph.nodes:
        if len(node.scope) > max_scope_len:
            max_scope_len = len(node.scope)
        if type(node) == Graph.Product:
            node_meta_info.append([0] + list(node.scope))
        elif type(node) == Graph.DistributionVector and len(list(einet.graph.successors(node))) == 0:
            node_meta_info.append([1] + list(node.scope))
        else:
            node_meta_info.append([2] + list(node.scope))
    
    # pad lists s.t. np.array works
    for idx in range(len(node_meta_info)):
        info = node_meta_info[idx]
        scope = info[1:]
        padding_len = max_scope_len - len(scope)
        if padding_len > 0:
            scope += [-1]*padding_len
        node_meta_info[idx] = [info[0]] + scope
        
    parameters = [p.cpu().detach().numpy() for p in einet.parameters() if p.requires_grad]

    return parameters + [adj] + [np.array(node_meta_info)]

def test(einet, data):
    """Computes log-likelihood in batched way."""
    with torch.no_grad():
        ll_total = 0.0
        for x, batch_labels in data:
            outputs = einet(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs, batch_labels)
            ll_total += ll_sample.sum().item()
        return ll_total
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu > -1 else torch.device('cpu')
    main('mnist', 2, args.id, device)