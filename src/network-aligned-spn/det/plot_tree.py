from tree import DensityTree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from itertools import product

np.random.seed(111)
data, labels = make_blobs(10000, 2, return_centers=False, centers=3)
data = np.column_stack([data, labels])

tree = DensityTree(4, ['cont', 'cont', 'cat'])
tree.train(data)

min_x, max_x = data[:, 0].min(), data[:, 0].max()
min_y, max_y = data[:, 1].min(), data[:, 1].max()

s = np.array(list(product(np.arange(min_x, max_x, 0.2), np.arange(min_y, max_y, 0.2), np.arange(0, 3))))

likelihoods = np.array([tree.predict(x) for x in s])

conditions = tree.get_conditions(tree.root_node)
print(conditions)

fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(s[:, 0], s[:, 1], c=abs(likelihoods), alpha=0.3, cmap='Greens')
plt.show()