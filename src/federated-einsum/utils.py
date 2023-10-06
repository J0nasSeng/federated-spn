import torch
import numpy as np
from torchvision.datasets import ImageNet
from torchvision.models import ViT_L_16_Weights
from kmeans_pytorch import kmeans
import math
import torch.nn.functional as F
from PIL import Image
import os
import errno

def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind

def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)

def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx

def extract_image_patches(x, size=4, stride=4):
    # x has shape [b, 3, 224, 224]
    patches = x.unfold(2, size, stride).unfold(3, size, stride)
    return patches

def patch_and_cluster_imagenet(root_clusters, imagenet_path, num_clusters=1000):
    """
        for each pre-computed cluster (can also be the labels), create 4x4 patches of each
        image. Then cluster images along the patch-dimension, i.e. we group images which have similar
        patches for each patch position.
        For each patch (we simply count them), store a dictionary mapping cluster-id to 
        the corresponding image ids and cluster means from that specific cluster.
        In the end we thus know which image-patch belongs to which cluster and can compare the
        cluster similarity to construct the Einsum structure.
    """
    transform = ViT_L_16_Weights.DEFAULT.transforms()
    imagenet = ImageNet(imagenet_path, transform=transform)

    patch_clusters_of_clusters = []

    for rc in np.unique(root_clusters):
        idx = np.argwhere(root_clusters == rc).flatten()
        subset = imagenet.imgs[idx]
        patches = extract_image_patches(subset, 8, 8)

        patch_clusters = {}

        for p in range(patches.shape[2]):
            patch_clusters[p] = {}
            # get p-th patch of all images and reshape to [B, C x P x S x S]
            # where B = number of images, C channels, P number of patches and S patch size 
            p_th_patches = patches[:, :, p].reshape((patches.shape[0], -1))

            # apply kmeans
            cluster_ids, cluster_means = kmeans(p_th_patches, num_clusters)

            # create store sample ids for each cluster of patch p
            for cid in torch.sort(torch.unique(cluster_ids)):
                # obtain ids of images for this patch-cluster
                cidx = torch.argwhere(cluster_ids == cid).flatten()
                patch_clusters[p][cid] = (cidx.numpy(), cluster_means[cid])

        patch_clusters_of_clusters.append(patch_clusters)
    return patch_clusters_of_clusters

def compare_patch_clusters(cluster_patch_clusters):
    """
        given a patch-cluster dict for each root cluster, compare the
        patch-clusters based on eucledian distance.
        Group similar clusters together -> will be the clusters connected via 
        a product node in SPN later on as similar clusters reduce heterogeinity.
    """
    for patch_clusters in cluster_patch_clusters:

        patch_clsts = []
        for p, clsts in patch_clusters.items():
            cluster_means = [cm for _, cm in clsts.values()]
            patch_clsts.append(cluster_means)
        
        # TODO: group clusters of patches s.t. similar clusters get in one group

def get_surrounding_patches(patches, i, j, device):
    """
        Given the patches of an image batch, return the surrounding patches of
        patch (i, j). If there is no patch (i.e. i=1 and/or j=1), return 0 instead.
    """

    patch_shape = patches[:, :, i, j].shape
    if (i - 1) == 0 and (j - 1) == 0:
        x_prev = [torch.zeros(patch_shape).to(device) for _ in range(3)]
    elif (i - 1) == 0 and (j - 1) > 0:
        x_prev = [torch.zeros(patch_shape).to(device), patches[:, :, i, j-1], torch.zeros(patch_shape).to(device)]
    elif (i - 1) > 0 and (j - 1) > 0:
        x_prev = [patches[:, :, i-1, j], torch.zeros(patch_shape).to(device), torch.zeros(patch_shape).to(device)]
    else:
        x_prev = [patches[:, :, i-1, j], patches[:, :, i, j-1], patches[:, :, i-1, j-1]]
    return x_prev

def set_einet_weights(einet, weights):
    """
        Given an Einet and the predicted parameters of a NN, set the 
        parameter of the Einet accordingly.
    """
    for w, p in zip(weights, einet.parameters()):
        assert w.shape == p.shape
        p.data = w
    return einet