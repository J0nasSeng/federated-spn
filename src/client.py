import flwr as fl
import torch
from torch.utils.data import DataLoader
from einsum import EinsumNetwork, Graph, EinetMixture
import config
from datasets import get_dataset_loader
from rtpt import RTPT
import networkx as nx
import argparse
import numpy as np
from utils import flwr_params_to_numpy, save_image_stack
import os
import logging
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main(dataset, num_clients, client_id, chk_dir, device):

    data_loader = get_dataset_loader(dataset, num_clients, config.dataset_inds_file, config.data_skew)
    train_data, val_data = data_loader.load_client_data(client_id)
    # artificially skew data 
    #if client_id == 0:
    #    idx = train_data.dataset.targets[train_data.dataset.targets == 0]
    #    train_data = Subset(train_data.dataset, idx)
    #else:
    #    idx = train_data.dataset.targets[train_data.dataset.targets == 1]
    #    train_data = Subset(train_data.dataset, idx)
    rtpt = RTPT('JS', 'FedSPN-Client', config.num_epochs)
    rtpt.start()

    class SPNClient(fl.client.NumPyClient):
        """
            This class implements a client which trains a SPN (locally).
            After training (one fit-call), it sends its learned parameters 
            to the server. 
        """

        def __init__(self):
            self.device = device
            self.train_loader = DataLoader(train_data, config.batch_size, True, num_workers=1, persistent_workers=True)
            self.val_loader = DataLoader(val_data, config.batch_size, True, num_workers=1, persistent_workers=True)
            self.einet = init_spn(device)

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
            mean = compute_dataset_mean(self.train_loader)
            self.einet = train(self.einet, self.train_loader, config.num_epochs, device, chk_dir, mean)
            samples = self.einet.sample(25, std_correction=0.0).detach().cpu().numpy()
            samples = samples.reshape((-1, config.height, config.width, config.num_dims))
            samples += mean.cpu().numpy() / 255.
            samples -= samples.min()
            samples /= samples.max()
            img_path = os.path.join(chk_dir, 'samples.png')
            save_image_stack(samples, 5, 5, img_path, margin_gray_val=0.)
            # collect parameters and send back to server
            params = self.get_parameters()
            # log final model
            torch.save(self.einet, os.path.join(chk_dir, 'chk_final.pt'))
            return params, len(train_data), {}
        
        def evaluate(self, parameters, cfg):
            """
                Evaluate the local SPN or global SPN (depending on what's sent by server)
            """
            einet = make_spn(parameters)
            # TODO: what to evaluate here?
            # for now, just save model
            samples = self.einet.sample(25)
            samples = samples.reshape((-1, 28, 28))
            save_image_stack(samples, 5, 5, os.path.join('../samples/demo_mnist/', "samples.png"), margin_gray_val=0.)
            torch.save(einet, './model.pt')

    # Start client
    fl.client.start_numpy_client(server_address="localhost:{}".format(config.port), client=SPNClient(),
                                 grpc_max_message_length=966368971)

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
        graph = Graph.poon_domingos_structure(shape=(config.height, config.width), delta=pd_delta, axes=[0, 1])
    elif config.structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=config.num_vars, depth=config.depth, num_repetitions=config.num_repetitions)
    elif config.structure == 'flat-binary-tree':
        graph = Graph.binary_tree_spn(shape=(config.height, config.width))
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=config.num_vars,
            num_dims=config.num_dims,
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

def compute_dataset_mean(train_loader):
    batch_means = []
    for x, _ in train_loader:
        x = x.permute((0, 2, 3, 1))
        batch_mean = torch.mean(x, dim=0)
        batch_means.append(batch_mean)
    batch_means = sum(batch_means) / len(batch_means)
    return batch_means

def train(einet, train_loader, num_epochs, device, chk_path, mean=None):

    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    logging.info('Starting Training...')
    for epoch_count in range(num_epochs):
        einet.train()

        if epoch_count > 0 and epoch_count % config.checkpoint_freq == 0:
            torch.save(einet, os.path.join(chk_path, f'chk_{epoch_count}.pt'))

        total_ll = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.permute((0, 2, 3, 1))
            if mean is not None:
                x -= mean
            x /= 255.
            x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
            x = x.to(device)
            outputs = einet.forward(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()

            if i % 20 == 0:
                logging.info('Epoch {:03d} \t Step {:03d} \t LL {:03f}'.format(epoch_count, i, total_ll))
        logging.info('Epoch {:03d} \t LL={:03f}'.format(epoch_count, total_ll))

        einet.em_update()
    return einet

def make_spn(params):
    """
        Given a parameter array, reconstruct the learnt SPN.
        
        params: List containing a description of the structure in form of
            a list of lists (=param[0]) and the SPN's parameters (=params[1])
    """
    num_clients = len(params) / 4
    einets = []
    component_params = []
    # reconstruct graphs based on adjacency, node-types and scopes
    # passed by client
    for it in range(num_clients):
        parameters = params[0]
        adj = params[1]
        meta_info = params[2]
        component_params = params[3]
        params = params[3:]
        graph = nx.DiGraph()
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
            num_dims=config.num_dims,
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
        einets.append(einet)

    mixture = EinetMixture.EinetMixture(component_params, einets)

    return mixture

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
    parser.add_argument('--checkpoint-dir', type=str)

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu > -1 else torch.device('cpu')
    main(config.dataset, config.num_clients, args.id, args.checkpoint_dir, device)