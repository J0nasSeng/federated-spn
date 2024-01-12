import numpy as np
from node import LeafNode, DecisionNode

class DensityTree:

    def __init__(self, max_depth, feature_types) -> None:
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.N = 0
        assert all([t in ['cont', 'ord', 'cat'] for t in feature_types])

    def predict(self, x):
        curr_node = self.root_node

        while not isinstance(curr_node, LeafNode):
            if curr_node.condition(x):
                # go left
                curr_node = curr_node.left
            else:
                #go right
                curr_node = curr_node.right
        
        return curr_node.density(self.N)

    def cost(self, data):
        """
            Compute the error according to the formula given in https://mlpack.org/papers/det.pdf
            It resembles the Integrated Square Error (ISE). 
        """
        mins = np.array([data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        maxs = np.array([data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        num_cats = np.array([len(np.unique(data[:, i])) for i in range(len(self.feature_types)) if self.feature_types[i] == 'cat'])
        ord_mins = np.array([data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])
        ord_maxs = np.array([data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])

        vol_rd = float(np.prod(maxs - mins))
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
            for i in range(data.shape[0]):
                ft = self.feature_types[j]
                if ft == 'cat':
                    left, right = data[data[:, j] == data[i, j]], data[data[:, j] != data[i, j]]
                else:
                    left, right = data[data[:, j] <= data[i, j]], data[data[:, j] > data[i, j]]

                if left.shape[0] < 2 or right.shape[0] < 2:
                    continue
                left_cost, right_cost = self.cost(left), self.cost(right)
                gain = curr_cost - left_cost - right_cost
                if gain > max_gain:
                    max_gain = gain
                    split_feat = j
                    val = data[i, j]
        print(curr_cost)
        print(max_gain)
        return split_feat, val

    def build_tree(self, data, depth=1):
        self.N = len(data)
        split_feat, val = self.find_cut(data)
        if split_feat == -1 or depth >= self.max_depth:
            leaf = LeafNode(data, depth, self.feature_types)
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
        self.root_node = self.build_tree(data, 1)

    def get_conditions(self, node):
        if isinstance(node, LeafNode):
            return
        if node.split_feature_type == 'cat':
            op = '=='
        else:
            op = '<='
        cond = str(node.split_feature) + op + str(node.split_val)
        return [cond, self.get_conditions(node.left), self.get_conditions(node.right)]