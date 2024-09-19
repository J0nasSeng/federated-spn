from tree import DensityTree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from itertools import product
from density_forest import DensityForest

NUM_CLASSES = 2
MODEL = 'tree' # choose between tree or forest
np.random.seed(111)
data, labels = make_blobs(10000, 2, return_centers=False, centers=NUM_CLASSES)
data = np.column_stack([data, labels])

if MODEL == 'tree':
    model = DensityTree(5, ['cont', 'cont', 'cat'], leaf_type='hist', num_bins=10, min_leaf_instances=20)
else:
    model = DensityForest(['cont', 'cont', 'cat'])
model.train(data)

min_x, max_x = data[:, 0].min(), data[:, 0].max()
min_y, max_y = data[:, 1].min(), data[:, 1].max()

s = np.array(list(product(np.arange(min_x, max_x, 0.2), np.arange(min_y, max_y, 0.2), np.arange(0, NUM_CLASSES))))

likelihoods = np.array([model.predict(x) for x in s])

# classify based on likelihood values
y_preds = []
for x in data:
    x[-1] = 1
    l1 = model.predict(x)
    x[-1] = 0
    l0 = model.predict(x)
    cl = np.argmax([l0, l1]).flatten()[0]
    y_preds.append(cl)
print(accuracy_score(labels, np.array(y_preds)))

fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(s[:, 0], s[:, 1], c=abs(likelihoods), alpha=0.3, cmap='Greens')
plt.colorbar()
plt.show()