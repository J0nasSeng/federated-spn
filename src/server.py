import flwr as fl
import config
from typing import Callable, Union, Optional, List, Tuple, Dict

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from einsum import EinsumNetwork, Graph
import networkx as nx

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

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
            As clients learn SPN locally, we don't initialize any parameters
        """
        return []

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
        # build global SPN
        spns = [make_spn(param) for param, _ in results]
        self.global_einet = make_global_spn(spns)
        spn_params = spn_to_param_list(self.global_einet)
        return spn_params, {}


def make_spn(params):
    """
        Given a parameter array, reconstruct the learnt SPN.
        
        params: List containing a description of the structure in form of
            a list of lists (=param[0]) and the SPN's parameters (=params[1])
    """
    structure, parameters = params

    # 0 = Sum node, 1 = Product node, 2 = Leaf
    _, root_scope = structure[0]
    root_node = Graph.DistributionVector(root_scope)

    graph = nx.DiGraph()
    graph.add_node(root_node)

    for layer in structure[1:]:
        # TODO: reconstruct graph by traversing list


def spn_to_param_list(einet: EinsumNetwork.EinsumNetwork):
    """
        Transform a EinsumNetwork object to a parameter-array containing
        the structure as a list of lists and the parameters as a separate array.
    """

    def traverse_graph(list_repr):
        layer = []
        for node in list_repr[-1]:
            succ_nodes = graph.successors(node)
            layer += list(succ_nodes)

        list_repr.append(layer)
        leaf_layer = [len(list(graph.successors(n))) == 0 for n in layer]
        if all(leaf_layer):
            return list_repr
        else:
            return traverse_graph(list_repr)

    graph = einet.graph
    root = Graph.get_roots(graph)[0]
    list_graph = traverse_graph([root])

    # apply enoding
    encoded_graph = []
    for layer in list_graph:
        encoded_layer = []
        for node in layer:
            # sum
            if type(node) == Graph.DistributionVector and len(list(graph.successors(node))) > 0:                
                encoded_layer.append((0, node.scope))
            elif type(node) == Graph.Product:
                encoded_layer.append((1, node.scope))
            else:
                encoded_layer.append((2, node.scope))
        encoded_graph.append(encoded_graph)
    
    parameters = [val.cpu().numpy() for _, val in einet.state_dict().items()]

    return [encoded_graph, parameters]


def main():

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=[],
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.communication_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()