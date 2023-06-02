from einsum import EinsumNetwork
import torch

#exponential_family = EinsumNetwork.BinomialArray
#exponential_family = EinsumNetwork.CategoricalArray
exponential_family = EinsumNetwork.NormalArray

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

classes = [7]
# classes = [2, 3, 5, 7]
# classes = None

K = 10

structure = 'poon-domingos'
# structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28
num_vars = width*height

# 'binary-trees'
depth = 3
num_repetitions = 20

num_epochs = 5
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

device = torch.device('cpu')

dataset_inds_file = 'indices.json'
data_skew = 0.0

# Server config
communication_rounds = 1

# Client config
port = '8080'