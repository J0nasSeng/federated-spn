from ceinsum import EinsumNetwork

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

K = 3

structure = 'poon-domingos'
#structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [2]
#pd_num_pieces = [7]
#pd_num_pieces = [8, 32]
width = 8
height = 8
num_vars = width*height
num_dims = 3

# 'binary-trees'
depth = 2
num_repetitions = 2

num_epochs = 10
batch_size = 10
online_em_frequency = 2
online_em_stepsize = 0.5
num_clients = 1

checkpoint_freq = 2

# Server config
communication_rounds = 1

# Client config
port = '12005'

# Devices
devices = [0, 1, 2, 3]
num_processes = 5