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
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

classes = [7]
# classes = [2, 3, 5, 7]
# classes = None

K = 10

structure = 'poon-domingos'
# structure = 'binary-trees'

# 'poon-domingos'
# pd_num_pieces = [4]
pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 224
height = 224
num_vars = width*height
num_dims = 3

# 'binary-trees'
depth = 3
num_repetitions = 20

num_epochs = 20
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05
num_clients = 25

checkpoint_freq = 2
checkpoint_dir = f'./checkpoints/chk_{round(time.time() * 1000)}/'

# copy config to checkpoint
os.makedirs(checkpoint_dir, exist_ok=True)
shutil.copyfile('./config.py', os.path.join(checkpoint_dir, 'config.py'))

dataset_inds_file = 'indices.json'
dataset = 'imagenet'
data_skew = 0.

# Server config
communication_rounds = 1

# Client config
port = '12000'