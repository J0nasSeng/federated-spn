from scipy.special import logsumexp

from spn.algorithms.Gradient import gradient_backward
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid

from spn.structure.Base import Sum, get_nodes_by_type, get_number_of_nodes
import numpy as np
from copy import deepcopy

def softmax(vec, temperature):
    """
    turn vec into normalized probability
    """
    sum_exp = sum(np.exp(x/temperature) for x in vec)
    return np.array([np.exp(x/temperature)/sum_exp for x in vec])

def cond_sum_em_update(allowed_nodes):
    def sum_em_update(node, node_gradients=None, root_lls=None, all_lls=None, **kwargs):
        if node.id in allowed_nodes:
            RinvGrad = node_gradients - root_lls

            for i, c in enumerate(node.children):
                new_w = RinvGrad + (all_lls[:, c.id] + np.log(node.weights[i]))
                node.weights[i] = logsumexp(new_w)

            assert not np.any(np.isnan(node.weights))

            node.weights = np.exp(node.weights - logsumexp(node.weights)) + np.exp(-100)

            node.weights = node.weights / node.weights.sum()
            #node.weights = softmax(node.weights, 0.1)
            idx = np.argsort(node.weights)[:-3]
            node.weights[idx] = 0
            node.weights = node.weights / node.weights.sum()


            if node.weights.sum() > 1:
                node.weights[np.argmax(node.weights)] -= node.weights.sum() - 1

            assert not np.any(np.isnan(node.weights))
            assert np.isclose(np.sum(node.weights), 1)
            assert not np.any(node.weights < 0)
            assert node.weights.sum() <= 1, "sum: {}, node weights: {}".format(node.weights.sum(), node.weights)
    return sum_em_update

_node_updates = {}

def add_node_em_update(node_type, lambda_func):
    _node_updates[node_type] = lambda_func


def EM_optimization_network(spn, data, iterations=5, node_updates=_node_updates, skip_validation=False, **kwargs):
    if not skip_validation:
        valid, err = is_valid(spn)
        assert valid, "invalid spn: " + err

    lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(spn)))

    for _ in range(iterations):
        # one pass bottom up evaluating the likelihoods
        log_likelihood(spn, data, dtype=data.dtype, lls_matrix=lls_per_node)

        gradients = gradient_backward(spn, lls_per_node)

        R = lls_per_node[:, 0]

        for node_type, func in node_updates.items():
            for node in get_nodes_by_type(spn, node_type):
                func(
                    node,
                    node_lls=lls_per_node[:, node.id],
                    node_gradients=gradients[:, node.id],
                    root_lls=R,
                    all_lls=lls_per_node,
                    all_gradients=gradients,
                    data=data,
                    **kwargs
                )