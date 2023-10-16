from conditional_feinsum import init_spn
import torch
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from utils import set_einet_weights, extract_image_patches, get_surrounding_patches
import numpy as np
from nns import MLP
import config
import torch.nn.functional as F
from PIL import Image
import argparse
from multiprocessing import Process

def eval_einsum(model_dir, model_id, dataset_dir, device_id):
    transform = Compose([ToTensor(), Resize(112), CenterCrop(112)])
    imagenet = ImageNet(dataset_dir, transform=transform, split='val')
    loader = DataLoader(imagenet, 64, num_workers=4)
    model_file = f'chk_{model_id}.pt'
    device = torch.device(f'cuda:{device_id}')
    einet = torch.load(model_dir + model_file).to(device)
    einet_lls = []
    for i, (x, y) in enumerate(loader):
        if i % 50 == 0:
            print(f"{(i / len(loader) * 100):3f}%")
        x = x.to(device)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
        ll_sample = einet.forward(x)
        einet_lls.append(ll_sample)
        print(ll_sample.shape)
    
    #einet_ll = einet_ll.numpy()
    #file = f'{model_dir}_eval_{model_id}'
    #np.savetxt(file, einet_ll)


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str)
parser.add_argument('--dataset-dir', type=str)
parser.add_argument('--cluster-file', type=str)
parser.add_argument('--max-processes', default=4, type=int)
parser.add_argument('--devices', nargs='+', type=int)

args = parser.parse_args()

if __name__ == '__main__':
    clusters = np.load(args.cluster_file)
    unique_clusters = np.unique(clusters)
    unique_clusters_batched = np.array_split(unique_clusters, args.max_processes)

    devices = args.devices

    for batch in unique_clusters_batched:
        processes = [] 
        for i, c in enumerate(batch):
            idx = i % len(devices)
            device_id = devices[idx]
            p = Process(target=eval_einsum, args=(args.model_dir, c, args.dataset_dir, device_id))

        for p in processes:
            p.join()