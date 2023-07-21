from einsum import EinsumNetwork
import time
import shutil
import os

#exponential_family = EinsumNetwork.BinomialArray
#exponential_family = EinsumNetwork.CategoricalArray
exponential_family = EinsumNetwork.NormalArray

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.01}

classes = [7]
# classes = [2, 3, 5, 7]
# classes = None

K = 30

structure = 'poon-domingos'
#structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
#pd_num_pieces = [7]
#pd_num_pieces = [8, 32]
width = 32
height = 32
num_vars = width*height
num_dims = 3

# 'binary-trees'
depth = 2
num_repetitions = 3

num_epochs = 10
batch_size = 32
online_em_frequency = 50
online_em_stepsize = 0.5
num_clients = 1

checkpoint_freq = 2

dataset_inds_file = 'indices.json'
dataset = 'svhn'
data_skew = 0.
grouping = 'cluster' # 'cluster'
num_classes = 10

num_clusters = 100
preprocessing = 'cluster', # 'mean'

# Server config
communication_rounds = 1

# Client config
port = '12005'