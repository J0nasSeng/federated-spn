import torch
import torch.nn as nn
from nns import MLP, CNNCondMixEin
import numpy as np
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.cluster import KMeans
from multiprocessing import Process
from rtpt import RTPT
from utils import save_image_stack, extract_image_patches, get_surrounding_patches
from einsum import EinsumNetwork, Graph
import config
import pickle
import sys
import os
import logging
from scipy.special import logsumexp
from einsum.EinsumNetwork import log_likelihoods
import argparse

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

class ConditionalMixtureEinsum:

    def __init__(self, ceinsum_mixtures, device, img_dims=(112, 112, 3), patch_dims=(14, 14)) -> None:
        """
            Instantiate a Conditional Mixture Einet.
            ceinsum_mixtures: I x J matrix of mixture models. I, J refers to number of vertical/horizontal patches of images
            param_nns: I x J neural networks taking surrounding patches of some patch (i, j) as input and yielding mixture weights
        """
        super().__init__()
        self.ceinsum_mixtures = ceinsum_mixtures
        self.num_patches = len(self.ceinsum_mixtures)
        self.height, self.width, self.channels = img_dims
        self.patch_height, self.patch_width = patch_dims
        self.device = device

        param_nns_match_ceinsums = len(ceinsum_mixtures) == len(ceinsum_mixtures[0])
        assert param_nns_match_ceinsums, 'Only square like image splits supported'

    def sample(self, N, y):
        """
            Sample from product distribution of Conditional Mixture Einsum
        """
        assert N == len(y), 'Need one label for each sample'
        samples = torch.zeros(N, self.channels, self.height, self.width).to(self.device)
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                ceinsum_mixture = self.ceinsum_mixtures[i][j]
                s = ceinsum_mixture.sample(samples, y, i, j)
                s = s.reshape(-1, self.channels, self.patch_height, self.patch_width)
                start_i, end_i = i*self.patch_height, (i+1)*self.patch_height
                start_j, end_j = j*self.patch_width, (j+1)*self.patch_width
                samples[:, :, start_i:end_i, start_j:end_j] = s
        return samples.permute(0, 2, 3, 1)

class EinetMixture:
    """A simple class for mixtures of Einets, implemented in numpy."""

    def __init__(self, p, einets, device, patch_size=config.patch_size):

        if len(p) != len(einets):
            raise AssertionError("p and einets must have the same length.")

        self.num_components = len(p)
        self.px, self.py = patch_size
        self.p = p
        self.einets = einets

        num_var = set([e.args.num_var for e in einets])
        if len(num_var) != 1:
            raise AssertionError("all EiNet components must have the same num_var.")
        self.num_var = list(num_var)[0]

        num_dims = set([e.args.num_dims for e in einets])
        if len(num_dims) != 1:
            raise AssertionError("all EiNet components must have the same num_dims.")
        self.num_dims = list(num_dims)[0]
        self.param_nn = CNNCondMixEin(len(p)).to(device)

    def sample(self, x, y, i, j, **kwargs):
        N = x.shape[0]
        # patch images and obtain 8x8 patches
        patches = extract_image_patches(x, self.px, self.py)
        x_prev = get_surrounding_patches(patches, i, j, x.device)
        # obtain einsum parameters
        params = self.param_nn(x_prev, y).detach().cpu()
        samples = torch.zeros(N, self.num_var, self.num_dims)
        for k in range(N):
            p = params[k].numpy()
            rand_idx = np.random.choice(np.arange(p.shape[0]), p=p)
            s = self.einets[rand_idx].sample(num_samples=1, **kwargs).squeeze().cpu()
            samples[k, ...] = s
        return samples

    def conditional_sample(self, x, marginalize_idx, **kwargs):
        marginalization_backup = []
        component_posterior = np.zeros((self.num_components, x.shape[0]))
        for einet_counter, einet in enumerate(self.einets):
            marginalization_backup.append(einet.get_marginalization_idx())
            einet.set_marginalization_idx(marginalize_idx)
            lls = einet.forward(x)
            lls = lls.sum(1)
            component_posterior[einet_counter, :] = lls.detach().cpu().numpy() + np.log(self.p[einet_counter])

        component_posterior = component_posterior - logsumexp(component_posterior, 0, keepdims=True)
        component_posterior = np.exp(component_posterior)

        samples = np.zeros((x.shape[0], self.num_var, self.num_dims))
        for test_idx in range(x.shape[0]):
            component_idx = np.argmax(component_posterior[:, test_idx])
            sample = self.einets[component_idx].sample(x=x[test_idx:test_idx + 1, :], **kwargs)
            samples[test_idx, ...] = sample.squeeze().cpu().numpy()

        # restore the original marginalization indices
        for einet_counter, einet in enumerate(self.einets):
            einet.set_marginalization_idx(marginalization_backup[einet_counter])

        return samples

    def log_likelihood(self, x, y, i, j):
        ll_total = 0.0
        # patch images and obtain 8x8 patches
        patches = extract_image_patches(x, self.px, self.py)
        x_in = patches[:, :, i, j]
        x_prev = get_surrounding_patches(patches, i, j, x.device)
        # obtain einsum parameters
        params = self.param_nn(x_prev, y)
        for sample in range(x_in.shape[0]):
            x_ = x_in[sample].unsqueeze(0)
            p = params[sample]

            x_ = x_.permute((0, 2, 3, 1))
            x_ = x_.reshape(x_.shape[0], config.num_vars, config.num_dims)
            
            lls = torch.zeros(1, self.num_components, device=x.device)
            for einet_count, einet in enumerate(self.einets):
                outputs = einet(x_)
                lls[:, einet_count] = log_likelihoods(outputs).squeeze()
                lls[:, einet_count] -= torch.log(p[einet_count])
            lls = torch.logsumexp(lls, dim=1)
            ll_total += lls.sum()
        return ll_total, params

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

def cluster_patches(x, y, patch_size=14, num_clusters=5):
    """
        Split each images into patches of size patch_size.
        Then flatten the patches and cluster them using KMeans.
        Returns a list of the following form:
        [[[dataset_patch_{i}_{j}_{k}]]] where (i, j) identifies
        patch of x and k is the cluster k obtained by KMeans in patch
        (i, j).

        x: preprocessed images of shape [b, 3, h, w] of some cluster
        y: labels of shape [b]
        patch_size: Size of patches used to split images
        num_clusters: Number of clusters for KMeans being applied on patches of x
    """
    patches = extract_image_patches(x, patch_size, patch_size)
    patch_datasets = [[None for _ in range(patches.shape[3])] for _ in range(patches.shape[2])]
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            x_ = patches[:, :, i, j]
            kmeans = KMeans(num_clusters)
            cluster_ids = kmeans.fit_predict(x_.reshape(x_.shape[0], -1).cpu().numpy())
            cluster_datasets = []
            for c in np.unique(cluster_ids):
                idx = np.argwhere(cluster_ids == c).flatten()
                cluster_x = x_[idx]
                cluster_y = y[idx]
                tds = TensorDataset(cluster_x, cluster_y)
                cluster_datasets.append(tds)
            patch_datasets[i][j] = cluster_datasets
    return patch_datasets

def train_einet(einet, train_loader, num_epochs, device):

    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    #logging.info('Starting Training...')
    for epoch_count in range(num_epochs):
        einet.train()

        total_ll = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            x = x.permute((0, 2, 3, 1))
            x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
            ll_sample = einet.forward(x)
            #ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()

            #if i % 20 == 0:
                #logging.info('Epoch {:03d} \t Step {:03d} \t LL {:03f}'.format(epoch_count, i, total_ll))
        total_ll = total_ll / (len(train_loader) * train_loader.batch_size)
        logging.info('Epoch {:03d} \t LL={:03f}'.format(epoch_count, total_ll))

        einet.em_update()
    return einet

def train_param_nn(mixture, train_loader, num_epochs, i, j, device):
    optimizer = torch.optim.Adam(mixture.param_nn.parameters(), 0.01)
    for ep in range(1, num_epochs+1):
        optimizer.zero_grad()

        total_ll = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            ll, params = mixture.log_likelihood(x, y, i, j)
            ll = -ll
            
            ll.backward()

            nn.utils.clip_grad.clip_grad_norm_(mixture.param_nn.parameters(), 1.)

            total_ll += ll.item()

            optimizer.step()
        logging.info('Epoch {:03d} \t LL={:03f}'.format(ep, total_ll))   
    return mixture 

def train_conditional_mixture_einsum(dataset, epochs, clusters, cid, device_id):
    device = torch.device(f'cuda:{device_id}')
    img_ids = np.argwhere(clusters == cid).flatten()

    subset = Subset(dataset, img_ids)
    full_loader = DataLoader(subset, 64)
    x, y = torch.cat([s for s, _ in full_loader], dim=0), torch.cat([s for _, s in full_loader], dim=0)
    patch_datasets = cluster_patches(x, y, config.patch_size[0], num_clusters=3)
    patches = [[i, j] for i in range(len(patch_datasets)) for j in range(len(patch_datasets[0]))]

    num_steps = len(patches) * len(patch_datasets[0][0])
    rt = RTPT('JS', 'FEinsum', num_steps)
    rt.start()

    # initialize I x J matrix for eients and parameter networks
    mixture_leafs = [[None]*len(patch_datasets)]*len(patch_datasets)
    
    for i, j in patches:
        cluster_datasets = patch_datasets[i][j]
        mixture_einets = []
        mixture_weights = []
        if not config.reuse_trained:
            for ds in cluster_datasets:
                mixture_weights.append(len(ds))
                loader = DataLoader(ds, batch_size=config.batch_size)
                einet = init_spn(device)
                einet = train_einet(einet, loader, config.num_epochs, device)
                mixture_einets.append(einet)
            p = np.array(mixture_weights) / np.sum(np.array(mixture_weights))
            mixture = EinetMixture(p, mixture_einets, device, config.patch_size)
            with open(f'./models/model_{i}_{j}', 'wb') as f:
                pickle.dump(mixture, f)
        else:
            with open(f'./models/model_{i}_{j}', 'rb') as f:
                mixture = pickle.load(f)
        mixture_leafs[i][j] = mixture
        rt.step()

    logging.info('Train Conditional Mixture Einsum')

    for i, j in patches:
        mixture = mixture_leafs[i][j]
        mixture = train_param_nn(mixture, full_loader, epochs, i, j, device=device)
        mixture_leafs[i][j] = mixture
        
    with open('./models/mixtures', 'wb') as f:
        pickle.dump(mixture_leafs, f)
    return mixture_leafs

def sample(N, mixture_leafs, device_id):
    device = torch.device(f'cuda:{device_id}')
    confeinsum = ConditionalMixtureEinsum(mixture_leafs, device)
    labels = torch.zeros(N).to(device)
    samples = confeinsum.sample(N, labels).cpu().numpy()
    img_path = os.path.join('./', 'samples_ceinsum.png')
    save_image_stack(samples, 5, 5, img_path, margin_gray_val=0., frame=2, frame_gray_val=0.0)


clusters = np.load('/storage-01/ml-jseng/imagenet-clusters/vit_cluster_minibatch_10K.npy')
encodings = np.load('/storage-01/ml-jseng/imagenet-clusters/vit_enc.npy')

transform = Compose([ToTensor(), Resize(112), CenterCrop(112)])
imagenet = ImageNet('/storage-01/datasets/imagenet/', transform=transform)

unique_clusters = [2400] #np.unique(clusters)
num_slices = int(np.ceil(len(unique_clusters) / config.num_processes))
unique_clusters = np.array_split(unique_clusters, num_slices)

parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true')

args = parser.parse_args()

# train einets in parallel. Start num_slices processes in parallel, wait
# until they finished and start next batch
if __name__ == '__main__':
    if args.sample:
       with open('./models/mixtures', 'rb') as f:
           mixtures = pickle.load(f)
           sample(25, mixtures, 0)         
    else:
        for cluster_batch in unique_clusters:
            processes = []
            for i, rc in enumerate(cluster_batch):
                mixtures = train_conditional_mixture_einsum(imagenet, 5, clusters, rc, 0)
                sample(25, mixtures, 0)