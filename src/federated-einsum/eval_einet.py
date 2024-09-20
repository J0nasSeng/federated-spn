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
from utils import save_image_stack
import os
from torchsummary import summary

def eval_einsum(model_dir, model_id, dataset_dir, sample_dir, device_id):
    transform = Compose([ToTensor(), Resize(112), CenterCrop(112)])
    imagenet = ImageNet(dataset_dir, transform=transform, split='val')
    loader = DataLoader(imagenet, 16, num_workers=2)
    model_file = f'chk_{model_id}.pt'
    device = torch.device(f'cuda:{device_id}')
    einet = torch.load(model_dir + model_file).to(device)
    summary(einet)
    einet_lls = []
    for i, (x, y) in enumerate(loader):
        if i % 50 == 0:
            print(f"{(i / len(loader) * 100):3f}%")
        x = x.to(device)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape(x.shape[0], config.num_vars, config.num_dims)
        ll_sample = einet.forward(x)
        einet_lls.append(ll_sample)
    
    #einet_ll = einet_ll.numpy()
    #file = f'{model_dir}_eval_{model_id}'
    #np.savetxt(file, einet_ll)
    
    samples = einet.sample(9).reshape(-1, config.height, config.width, 3)
    save_image_stack(samples.cpu(), 3, 3, os.path.join(sample_dir, 'samples.png'))

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str)
parser.add_argument('--dataset-dir', type=str)
parser.add_argument('--device', type=int)
parser.add_argument('--sample-dir', default='./samples/')

args = parser.parse_args()

if __name__ == '__main__':

    device = args.device

    eval_einsum(args.model_dir, 0, args.dataset_dir, args.sample_dir, args.device)