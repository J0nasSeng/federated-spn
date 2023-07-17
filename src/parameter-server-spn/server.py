import flwr as fl
import config
from typing import Union, Optional, List, Tuple, Dict
import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from einsum import EinsumNetwork, Graph, EinetMixture
import networkx as nx
from datasets import get_dataset_loader
import numpy as np
from utils import flwr_params_to_numpy, save_image_stack, mkdir_p
import os
import networkx as nx
from rtpt import RTPT
import time

class FedSPNStrategy(fl.server.strategy.Strategy):

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = config.num_clients,
        min_evaluate_clients: int = config.num_clients,
        min_available_clients: int = config.num_clients,
        sample_dir: str = './samples/',
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        loader = get_dataset_loader(config.dataset, config.num_clients, config.dataset_inds_file, config.data_skew)
        loader.partition()

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
            As clients learn SPN locally, we don't initialize any parameters
        """
        return fl.common.ndarrays_to_parameters(np.array([]))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        results = [
            (fit_res.parameters, fit_res.num_examples)
            for _, fit_res in results
        ]

        num_exp = [ne for _, ne in results]
        # build global SPN     
        spns = [make_spn(param) for param, _ in results]
        p = [ne/sum(num_exp) for ne in num_exp]
        mixture = EinetMixture.EinetMixture(p, spns)
        samples = mixture.sample(25)
        samples = samples.reshape((-1, 28, 28))
        img_path = os.path.join(sample_dir, 'samples.png')
        save_image_stack(samples, 5, 5, img_path, margin_gray_val=0.)
        spn_params = spn_to_param_list(mixture)
        return spn_params, {}
    
    def evaluate(self, server_round, parameters):
        return 0, {}

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        return 0, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def configure_fit(self, server_round, parameters, client_manager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        fit_configurations = []
        for client in clients:
            fit_configurations.append((client, FitIns(parameters, {})))
        return fit_configurations
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

def make_spn(params):
    """
        Given a parameter array, reconstruct the learnt SPN.
        
        params: List containing a description of the structure in form of
            a list of lists (=param[0]) and the SPN's parameters (=params[1])
    """
    parameters, adj, meta_info = flwr_params_to_numpy(params)

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

    return einet


def spn_to_param_list(mixture: EinetMixture.EinetMixture):
    """
        Transform a EinsumNetwork object to a parameter-array containing
        the structure as a list of lists and the parameters as a separate array.
    """

    parameters = []
    max_scope_len = 0
    component_params = list(mixture.p)
    for einet in mixture.einets:
        adj = nx.convert_matrix.to_numpy_array(einet.graph)
        node_meta_info = []
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
            
        params = [p.cpu().detach().numpy() for p in einet.parameters() if p.requires_grad]
        parameters.append(params)
        parameters.append(adj)
        parameters.append(np.array(node_meta_info))
        parameters.append(component_params)
    
    return parameters

def main(sample_dir):

    # Create strategy
    strategy = FedSPNStrategy(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        sample_dir=sample_dir,
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address=f"localhost:{config.port}",
        config=fl.server.ServerConfig(num_rounds=config.communication_rounds),
        strategy=strategy,
        grpc_max_message_length=966368971
    )


if __name__ == "__main__":

    run_id = str(round(time.time() * 1000))

    sample_dir = f'./samples/{run_id}'
    os.makedirs(sample_dir, exist_ok=True)

    rt = RTPT('JS', 'FedSPN Server', 1)
    rt.start()
    main(sample_dir)
    rt.step()