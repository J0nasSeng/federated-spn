from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type
from spn.algorithms.Inference import add_node_likelihood
import numpy as  np

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
    return data[:, node.scope]

add_node_likelihood(SPNLeaf, lambda_func=forward_ll)