import numpy as np

class DecisionNode:
    def __init__(self, depth, split_feature, split_feature_type, split_val) -> None:
        self.depth = depth
        self.split_feature = split_feature
        self.split_feature_type = split_feature_type
        self.split_val = split_val
        self.left: LeafNode | DecisionNode
        self.right: LeafNode | DecisionNode

    def condition(self, x):
        if self.split_feature_type == 'cat':
            return x[self.split_feature] == self.split_val
        else:
            return x[self.split_feature] <= self.split_val

class LeafNode:

    def __init__(self, data, depth, feature_types) -> None:
        self.data = data
        self.depth = depth
        self.feature_types = feature_types

    def density(self, N):
        """
            Compute (piecewise constant) denisty of a certain region of the feature space represented by this node.
            Follows approach of https://mlpack.org/papers/det.pdf
            NOTE: Currently not exact, requires decision path for continuous features
            TODO: Think about making exact if performance not good enough
        """

        mins = np.array([self.data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        maxs = np.array([self.data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'cont'])
        num_cats = np.array([len(np.unique(self.data[:, i])) for i in range(len(self.feature_types)) if self.feature_types[i] == 'cat'])
        ord_mins = np.array([self.data[:, i].min() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])
        ord_maxs = np.array([self.data[:, i].max() for i in range(len(self.feature_types)) if self.feature_types[i] == 'ord'])

        vol_rd = np.prod(maxs - mins)
        prod_cats = np.prod(num_cats)
        ord_vol = np.prod(ord_maxs - ord_mins)

        return len(self.data) / (N * vol_rd * prod_cats * ord_vol)