from tree import DensityTree
import numpy as np

class DensityForest:

    def __init__(self, feature_types, num_estimators=20, subset_frac=0.6, max_depth=4) -> None:
        self.num_estimators = num_estimators
        self.subset_frac = subset_frac
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.trees = []

    def train(self, data):
        for _ in range(self.num_estimators):
            rnd_idx = np.random.choice(np.arange(len(data)), size=int(len(data) * self.subset_frac))
            subset = data[rnd_idx]
            tree = DensityTree(self.max_depth, self.feature_types, leaf_type='hist')
            tree.train(subset)
            self.trees.append(tree)
        
    def predict(self, x):
        preds = np.array([t.predict(x) for t in self.trees])
        return np.mean(preds)
