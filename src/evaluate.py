from einsum.EinetMixture import EinetMixture
import torch
import argparse
import os
from utils import save_image_stack
import config

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument('--checkpoint-name', type=str, default='chk_final.pt')
parser.add_argument('--sample-dir', type=str)

args = parser.parse_args()

client_einets = []
for dir in os.listdir(args.checkpoint_dir):
    if dir.startswith('client'):
        model = torch.load(os.path.join(args.checkpoint_dir, dir, args.checkpoint_name))
        client_einets.append(model)
        break

p = [1/len(client_einets)] * len(client_einets)
mixture = EinetMixture(p, client_einets)

samples = mixture.sample(25, std_correction=0.0)
samples = samples.reshape((-1, config.height, config.width, config.num_dims))
img_path = os.path.join(args.sample_dir, 'samples.png')
save_image_stack(samples, 5, 5, img_path, margin_gray_val=0., frame=2)