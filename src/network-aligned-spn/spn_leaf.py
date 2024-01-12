from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type
from spn.algorithms.Inference import add_node_likelihood
import numpy as  np
import torch

class DensityLeaf(Leaf):

    def __init__(self, model, scope=None):
        super().__init__(scope)

    @property
    def type(self):
        return Type.REAL
    
def forward_ll(node, data=None, **kwargs):
    ll = node.model.predict(data[:, node.scope])
    return np.exp(ll)

add_node_likelihood(DensityLeaf, lambda_func=forward_ll)

class SPNLeaf(Leaf):

    def __init__(self, scope=None):
        super().__init__(scope)

    @property
    def type(self):
        return Type.REAL
    

def forward_ll(node, data=None, **kwargs):
    """
        data = log likelihood of sub-spns -> just return
    """
    return np.exp(data[:, node.scope])

add_node_likelihood(SPNLeaf, lambda_func=forward_ll)

class SPNEinsumLeaf(Leaf):

    def __init__(self, einsum, scope=None):
        super().__init__(scope)
        self.einsum = einsum

    @property
    def type(self):
        return Type.REAL
    

def forward_ll(node, data=None, **kwargs):
    """
        data = log likelihood of sub-spns -> just return
    """
    return torch.exp(data[:, node.scope])

add_node_likelihood(SPNEinsumLeaf, lambda_func=forward_ll)