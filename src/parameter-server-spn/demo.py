import os
import numpy as np
import torch
from einsum import Graph, EinsumNetwork
import datasets
import utils
import config
from torch.utils.data import DataLoader

device = torch.device(f'cuda:{1}')

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 
"""
print(demo_text)

############################################################################
fashion_mnist = False

#exponential_family = EinsumNetwork.BinomialArray
# exponential_family = EinsumNetwork.CategoricalArray
exponential_family = EinsumNetwork.NormalArray

classes = [7]
# classes = [2, 3, 5, 7]
# classes = None

K = 40

structure = 'poon-domingos'
# structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 32
height = 32

# 'binary-trees'
depth = 3
num_repetitions = 20

num_epochs = 3
batch_size = 10
online_em_frequency = 50
online_em_stepsize = 0.5
############################################################################

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.01}

# get data
data = datasets.get_dataset_loader('svhn', 1, config.dataset_inds_file, config.data_skew)
train_data, val_data = data.load_client_data(0)
train_loader, val_loader = DataLoader(train_data, 32), DataLoader(val_data, 32)

# Make EinsumNetwork
######################################
pd_delta = [[height / d, width / d] for d in pd_num_pieces]
graph = Graph.poon_domingos_structure(shape=(height, width), axes=[1], delta=[8])
args = EinsumNetwork.Args(
        num_var=32*32,
        num_dims=3,
        num_classes=1,
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

# Train
######################################

train_N = len(train_data)
valid_N = len(val_data)

for epoch_count in range(num_epochs):

    ##### evaluate
    #einet.eval()
    #train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    #valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    #test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    #print("[{}]   train LL {}   valid LL {}   test LL {}".format(
    #    epoch_count,
    #    train_ll / train_N,
    #    valid_ll / valid_N,
    #    test_ll / test_N))
    einet.train()
    #####

    total_ll = 0.0
    for x, y in train_loader:
        idx = torch.argwhere(y == 0)
        x = x[idx]
        batch_x = x.reshape((-1, 32*32, 3)).to(device)
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()

model_dir = '../models/einet/demo_svhn/'
samples_dir = '../samples/demo_svhn/'
utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)

#####################
# draw some samples #
#####################

samples = einet.sample(num_samples=25).cpu().numpy()
samples = samples.reshape((-1, 32, 32, 3))
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "samples.png"), margin_gray_val=0.)