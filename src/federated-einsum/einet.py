import numpy as np
from torchvision.datasets import ImageNet, SVHN
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from einsum import EinsumNetwork, Graph, EinetMixture
import config
import torch
import os
import logging
import sys
from utils import save_image_stack
from sklearn.cluster import KMeans
from multiprocessing import Process
import pickle
from rtpt import RTPT
import pandas as pd

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

def init_spn(device):
    """
        Build a SPN (implemented as an einsum network). The structure is either
        the same as proposed in https://arxiv.org/pdf/1202.3732.pdf (referred to as
        poon-domingos) or a binary tree.

        In case of poon-domingos the image is split into smaller hypercubes (i.e. a set of
        neighbored pixels) where each pixel is a random variable. These hypercubes are split further
        until we operate on pixel-level. The spplitting is done randomly. For more information
        refer to the link above.
    """
    logging.info("Init Einsum...")
    if config.structure == 'poon-domingos':
        pd_delta = [[config.height / d, config.width / d] for d in config.pd_num_pieces]
        graph = Graph.poon_domingos_structure(shape=(config.height, config.width), delta=pd_delta)
    elif config.structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=config.num_vars, depth=config.depth, num_repetitions=config.num_repetitions)
    elif config.structure == 'flat-binary-tree':
        graph = Graph.binary_tree_spn(shape=(config.height, config.width))
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=config.num_vars,
            num_dims=config.num_dims,
            num_classes=1,
            num_sums=config.K,
            num_input_distributions=config.K,
            exponential_family=config.exponential_family,
            exponential_family_args=config.exponential_family_args,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    return einet

def train(num_epochs, device_id, chk_path, cluster_count):

    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    rt = RTPT('JS', 'FedEinsum', num_epochs)
    rt.start()
    logging.info('Starting Training...')
    log_likelihoods = []
    device = torch.device(f'cuda:{device_id}')
    # reserve GPU
    tensor = torch.randn(3).to(device)
    transform = Compose([ToTensor(), Resize(config.height, antialias=True), CenterCrop(config.height)])
    #dataset = SVHN('../datasets/', transform=transform, download=True)
    dataset = ImageNet('/storage-01/datasets/imagenet/', transform=transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2)
    einet = init_spn(device)
    for epoch_count in range(num_epochs):
        einet.train()

        total_ll = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            x = x.permute((0, 2, 3, 1))
            x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
            ll_sample = einet.forward(x)
            #ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item() / (len(loader) * loader.batch_size)

            if i % 20 == 0:
                logging.info('Epoch {:03d} \t Step {:03d} \t LL {:03f}'.format(epoch_count, i, total_ll))
        #total_ll = total_ll / (len(loader) * loader.batch_size)
        log_likelihoods.append(total_ll)
        logging.info('Epoch {:03d} \t LL={:03f}'.format(epoch_count, total_ll))

        einet.em_update()
        rt.step()
    torch.save(einet, os.path.join(chk_path, f'chk_{cluster_count}.pt'))
    df = pd.DataFrame(data=log_likelihoods, columns=['lls'])
    df.to_csv(os.path.join(chk_path, f'chk_{cluster_count}.csv'))
    return einet

if __name__ == '__main__':
    train(config.num_epochs, config.devices[0], './checkpoints_1gpus_1procs_v1/', 0)


#weights = np.array(cluster_sizes) / np.sum(cluster_sizes)
#cluster_idx = np.random.choice(np.arange(len(weights)), 3, p=weights)
#for cidx in cluster_idx:
#    samples = mixture.sample(100)
#    t_samples = torch.from_numpy(samples).to(device=device, dtype=torch.float32)
#    ll = torch.tensor([mixture.log_likelihood(t_samples[i].unsqueeze(0)) for i in range(len(samples))])
#    best_25, inds = torch.sort(ll, descending=True)
#    inds = inds.numpy()
#    samples = samples[inds[:25]]
#    samples = samples.reshape(-1, config.height, config.width, config.num_dims)
#    img_path = os.path.join('./', f'samples_{cidx}.png')
#    save_image_stack(samples, 5, 5, img_path, margin_gray_val=0., frame=2, frame_gray_val=0.0)

# show some images from some clusters
#rand_clusters = [2400] #np.random.randint(0, clusters.max(), size=50)
#for rc in np.unique(rand_clusters):
#    img_ids = np.argwhere(clusters == rc).flatten()
#    cluster_sizes.append(len(img_ids))
#    print(f"CLUSTER_SIZE={len(img_ids)}")
#    print(f"CLUSTER={rc}")
#    
#    subset = Subset(imagenet, img_ids)
#    loader = DataLoader(subset, batch_size=config.batch_size)
#    device = torch.device(f'cuda:{0}')
#    einet = init_spn(device)
#    einet = train(einet, loader, config.num_epochs, device, './checkpoints/', save_model=False)
#    root_einets.append(einet)
#
#weights = np.array(cluster_sizes) / np.sum(cluster_sizes)
#mixture = EinetMixture.EinetMixture(weights, root_einets)
#root_einets.append(mixture)
#
#samples = mixture.sample(25)
#samples = samples.reshape(-1, config.height, config.width, config.num_dims)
#img_path = os.path.join('./', f'samples.png')
#save_image_stack(samples, 5, 5, img_path, margin_gray_val=0., frame=2, frame_gray_val=0.0)