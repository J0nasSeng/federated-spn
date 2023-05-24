import flwr as fl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from einsum import EinsumNetwork, Graph
import config
from datasets import get_dataset_loader
from rtpt import RTPT

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
            self.einet = make_spn()

        def set_parameters(self, parameters):
            """
                Set parameters of SPN.
            """
        
        def get_parameters(self):
            """
                Get SPN parameters
            """
            return [val.cpu().numpy() for _, val in self.einet.state_dict().items()]

        def fit(self, parameters, cfg):
            """
                Fit SPN and send parameters to server
            """
            self.einet = train(self.einet, self.train_loader, config.num_epochs, config.device)

            # collect parameters and send back to server
            params = self.get_parameters()
            return params, len(train_data), {}
        
        def evaluate(self, parameters, cfg):
            """
                Evaluate the local SPN or global SPN (depending on what's sent by server)
            """
            eval_local = cfg['eval']['eval_local']
            if eval_local:
                # evaluate local SPN
                ll = test(self.einet, self.train_loader)
                return ll, len(train_data), {'log-likelihood': ll}
            else:
                # build and evaluate global SPN
                # TODO: Implement this!
                return
    # Start client
    fl.client.start_numpy_client("[::]:{}".format(config.port), client=SPNClient())

def make_spn(device):
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

def test(einet, data):
    """Computes log-likelihood in batched way."""
    with torch.no_grad():
        ll_total = 0.0
        for x, batch_labels in data:
            outputs = einet(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs, batch_labels)
            ll_total += ll_sample.sum().item()
        return ll_total