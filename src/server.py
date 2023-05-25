import flwr as fl
import config
from typing import Union, Optional, List, Tuple, Dict
from collections import OrderedDict
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from einsum import EinsumNetwork, Graph
import networkx as nx
from datasets import get_dataset_loader
import numpy as np
from utils import flwr_params_to_numpy

class FedSPNStrategy(fl.server.strategy.Strategy):

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        loader = get_dataset_loader('mnist', 2, config.dataset_inds_file, config.data_skew)
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
        print(failures)

        results = [
            (fit_res.parameters, fit_res.num_examples)
            for _, fit_res in results
        ]

        # build global SPN     
        spns = [make_spn(param) for param, _ in results]
        self.global_einet = make_global_spn(spns)
        spn_params = spn_to_param_list(self.global_einet)
        return ndarrays_to_parameters(spn_params), {}
    
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


def make_global_spn(spns: List[EinsumNetwork.EinsumNetwork]):
    # 1. merge graphs
    new_graph = merge_graphs(spns)

    # 2. create new spn using merged graph
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
    
    einet = EinsumNetwork.EinsumNetwork(new_graph, args)
    # TODO: replace initialize with setting the correct client parameters.
    #   still not clear how to do (especially in the first layer)
    einet.initialize()
    for p in einet.parameters():
        print(p.shape)
    return einet

def merge_graphs(spns: List[EinsumNetwork.EinsumNetwork]):
    """
        Adds new root node (sum node) and merges the SPN graphs from
        all clients into one single graph
    """
    graphs = [spn.graph for spn in spns]
    merged = graphs[0]
    nodes = list(merged.nodes)
    root = nodes[0]
    for g in graphs[1:]:
        # remember all successors of g's root
        groot = list(g.nodes)[0]
        groot_succ = g.successors(groot)
        # add all nodes and edges from g to merged
        g_nodes = list(g.nodes)[1:] # all except root
        merged.add_nodes_from(g_nodes)
        g_edges = [(n1, n2) for n1, n2 in g.edges if n1 != groot]
        merged.add_edges_from(g_edges)

        # connect root with successors of g's root (g's root gets dropped)
        for succ in groot_succ:
            merged.add_edge(root, succ)
    
    return merged

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

def main():

    # Create strategy
    strategy = FedSPNStrategy(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address=f"[::]:{config.port}",
        config=fl.server.ServerConfig(num_rounds=config.communication_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()