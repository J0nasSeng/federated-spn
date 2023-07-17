from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from datasets import get_dataset_loader
import torch
import os
import numpy as np
import pickle
import torchvision

def compute_cluster_means(data, cluster_idx, file):
    if os.path.exists(f'./precomputed/{file}'):
        batch_means = np.loadtxt(f'./precomputed/{file}')
        return batch_means
    unique_idx = np.unique(cluster_idx)
    means = np.zeros((len(unique_idx), 224, 224, 3), dtype=np.float32)
    for k in unique_idx:
        means[k, ...] = np.mean(data[cluster_idx == k, ...].astype(np.float32), 0)
    np.save(f'./precomputed/{file}', means)
    return means

def cluster_data(train_loader, file):
    train_data = torch.concat([x.permute((0, 2, 3, 1)) for x, _ in train_loader]).numpy()
    if os.path.exists(f'./precomputed/clusters/{file}'):
        means, idx = pickle.load(open(f'./precomputed/clusters/{file}', 'rb'))
        return means, idx, train_data
    # path does not exist -> create
    os.makedirs(f'./precomputed/clusters/', exist_ok=True)
    kmeans = MiniBatchKMeans(n_clusters=100,
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
    pickle.dump((means, idx), open(f'./precomputed/clusters/{file}', 'wb'))
    return means, idx, train_data

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                            torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0,)*3, std=(1/255,)*3)])
train_data = torchvision.datasets.ImageNet('~/datasets/imagenet', split='train', transform=transform)
train_loader = DataLoader(train_data, 128)

loader = get_dataset_loader('imagenet', 15, 'indices.json', 0)
loader.partition()
print("Clustering...")
for c in range(15):
    train_data, val_data = loader.load_client_data(c)
    train_loader = DataLoader(train_data, 128)
    val_data = DataLoader(val_data, 128)
    means, idx, train_x = cluster_data(train_loader, f'cluster_{c}')
    print(f"Compute means of client {c}...")
    means = compute_cluster_means(train_x, idx, f'mean_{c}')
    print("Done!")