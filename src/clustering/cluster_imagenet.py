from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans

import torch
import os
import numpy as np
import pickle
import torchvision

def compute_cluster_means(data, cluster_idx):
    if os.path.exists(f'./precomputed/means'):
        batch_means = np.loadtxt(f'./precomputed/means')
        return batch_means
    unique_idx = np.unique(cluster_idx)
    means = np.zeros((len(unique_idx), 224, 224, 3), dtype=np.float32)
    for k in unique_idx:
        means[k, ...] = np.mean(data[cluster_idx == k, ...].astype(np.float32), 0)
    np.save(f'./precomputed/means', means)
    return means

def cluster_data(train_loader):
    train_data = torch.concat([x.permute((0, 2, 3, 1)) for x, _ in train_loader]).numpy()
    if os.path.exists(f'./precomputed/clusters/cluster'):
        means, idx = pickle.load(open(f'./precomputed/clusters/cluster', 'rb'))
        return means, idx, train_data
    # path does not exist -> create
    os.makedirs(f'./precomputed/clusters/', exist_ok=True)
    kmeans = MiniBatchKMeans(n_clusters=1000,
                                max_iter=100,
                                n_init=3,
                                verbose=3,
                                batch_size=1024).fit(train_data.reshape(train_data.shape[0], -1))
    #kmeans = KMeans(n_clusters=config.num_clusters,
    #                verbose=3,
    #                max_iter=100,
    #                n_init=3).fit(cluster_dataset.reshape(cluster_dataset.shape[0], -1))
    means = kmeans.cluster_centers_
    idx = kmeans.labels_
    #aidx = kmeans.predict(train_data.reshape(train_data.shape[0], -1))
    pickle.dump((means, idx), open(f'./precomputed/clusters/cluster', 'wb'))
    return means, idx, train_data

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                            torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
train_data = torchvision.datasets.ImageNet('~/datasets/imagenet', split='train', transform=transform)
train_loader = DataLoader(train_data, 128)

print("Clustering...")
means, idx, train_x = cluster_data(train_loader)
print("Compute means...")
means = compute_cluster_means(train_x, idx)
print("Done!")