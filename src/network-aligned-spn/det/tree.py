import numpy as np
from . import LeafNode, DecisionNode, PCLeafNode, NonParamLeafNode, HistLeafNode
import networkx as nx
from .utils import hierarchy_pos

class DensityTree:

    def __init__(self, max_depth, feature_types, 
                 leaf_type='hist', min_leaf_instances=2,
                 num_bins=5, scope=None) -> None:
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.N = 0
        self.leaf_type = leaf_type
        self.min_leaf_instances = min_leaf_instances
        self.num_bins = num_bins
        self.scope = scope # important for Federated Circuits
        self.data: np.ndarray
        assert all([t in ['cont', 'ord', 'cat'] for t in feature_types])

    def predict(self, x):
        curr_node = self.root_node

        while not (isinstance(curr_node, NonParamLeafNode) or isinstance(curr_node, PCLeafNode)
                   or isinstance(curr_node, LeafNode) or isinstance(curr_node, HistLeafNode)):
            if curr_node.condition(x):
                # go left
                curr_node = curr_node.left
            else:
                #go right
                curr_node = curr_node.right
        density = curr_node.density(x, self.N)
        return density

    def cost(self, data):
        """
            Compute the error according to the formula given in https://mlpack.org/papers/det.pdf
            It resembles the Integrated Square Error (ISE). 
        """
        mins = np.array([data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        maxs = np.array([data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont']) + 1e-6 # ensure non-zero difference
        num_cats = np.array([len(np.unique(data[:, i])) for i in range(len(self.feature_types)) if self.feature_types[i] == 'cat'])
        ord_mins = np.array([data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])
        ord_maxs = np.array([data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])

        vol_rd = np.prod(maxs - mins).item()
        prod_cats = float(np.prod(num_cats))
        ord_vol = float(np.prod(ord_maxs - ord_mins))
        error = - ((len(data)**2) / (self.N**2 * vol_rd * prod_cats * ord_vol))
        return error

    def find_cut(self, data):
        max_gain = 0
        curr_cost = self.cost(data)
        split_feat = -1
        val = -1

        for j in range(data.shape[1]):
            ft = self.feature_types[j]
            feature = data[:, j]
            ufeature = np.unique(feature)
            for u in ufeature:
                if ft == 'cat':
                    left, right = data[data[:, j] == u], data[data[:, j] != u]
                else:
                    left, right = data[data[:, j] <= u], data[data[:, j] > u]

                if left.shape[0] < self.min_leaf_instances or right.shape[0] < self.min_leaf_instances:
                    continue
                left_cost, right_cost = self.cost(left), self.cost(right)
                gain = curr_cost - left_cost - right_cost
                if gain > max_gain:
                    max_gain = gain
                    split_feat = j
                    val = u

        return split_feat, val

    def build_tree(self, data, depth=1):
        self.N = len(data)
        split_feat, val = self.find_cut(data)
        if split_feat == -1 or depth >= self.max_depth:
            if self.leaf_type == 'non-param':
                leaf = NonParamLeafNode(data, depth, self.feature_types)
            elif self.leaf_type == 'pc':
                leaf = PCLeafNode(data, depth, self.feature_types)
            elif self.leaf_type == 'hist':
                leaf = HistLeafNode(data, depth, self.feature_types, self.num_bins)
            else:
                raise ValueError(f'No such leaf type: {self.leaf_type}')
            return leaf

        node = DecisionNode(depth, split_feat, self.feature_types[split_feat], val)
        ft = self.feature_types[split_feat]
        if ft == 'cat':
            left_data, right_data = data[data[:, split_feat] == val], data[data[:, split_feat] != val]
        else:
            left_data, right_data = data[data[:, split_feat] <= val], data[data[:, split_feat] > val]
        node.left = self.build_tree(left_data, depth + 1)
        node.right = self.build_tree(right_data, depth + 1)
        return node
    
    def train(self, data):
        self.data = data
        self.root_node = self.build_tree(data, 1)

    def _build_nx_tree(self, nodes, graph: nx.DiGraph):
        next_nodes = []
        for n in nodes:
            if not issubclass(type(n), LeafNode):
                l, r = n.left, n.right
                if n.split_feature_type== 'cont':
                    n_repr = f'{n.split_feature} <= {round(n.split_val, 3)}'
                else:
                    n_repr = f'{n.split_feature} == {n.split_val}'
                if issubclass(type(l), LeafNode):
                    l_repr = l
                else:
                    if l.split_feature_type == 'cont':
                        l_repr = f'{l.split_feature} <= {round(l.split_val, 3)}'
                    else:
                        l_repr = f'{l.split_feature} == {l.split_val}'
                    next_nodes.append(l)
                graph.add_edge(n_repr, l_repr)
                if issubclass(type(r), LeafNode):
                    r_repr = r
                else:
                    if l.split_feature_type == 'cont':
                        r_repr = f'{r.split_feature} <= {round(r.split_val, 3)}'
                    else:
                        r_repr = f'{r.split_feature} == {r.split_val}'
                    next_nodes.append(r)
                graph.add_edge(n_repr, r_repr)
        if len(next_nodes) == 0:
            return graph
        return self._build_nx_tree(next_nodes, graph)

    def visualize(self):
        graph = nx.DiGraph()
        self._build_nx_tree([self.root_node], graph)
        pos = hierarchy_pos(graph)
        nx.draw(graph, pos, with_labels=True)

    def get_conditions(self, node):
        if issubclass(type(node), LeafNode):
            return
        if node.split_feature_type == 'cat':
            op = '=='
        else:
            op = '<='
        cond = str(node.split_feature) + op + str(node.split_val)
        return [cond, self.get_conditions(node.left), self.get_conditions(node.right)]