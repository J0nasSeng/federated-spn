from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from datasets import get_dataset_loader
import torch
import torch.nn as nn
import rtpt
import numpy as np
import torchvision
from torchvision.models import resnet152, vit_l_16, ResNet152_Weights, ViT_L_16_Weights
import argparse
from kmeans_pytorch import kmeans

def load_pretrained(model_name):
    if model_name == 'resnet':
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity() # throw away classification head as we want to encode features
        return model
    elif model_name == 'vit':
        model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        return model
        
    
def encode_imagenet(args):
    """
        Given a pretrained model, encode imagenet data and
        store the encodings ready for subsequent clustering
    """
    print(f"Encoding using {args.model}")
    model = load_pretrained(args.model)
    model = model.to(f'cuda:{args.gpu}')
    if args.model == 'resnet':
        transform = ResNet152_Weights.DEFAULT.transforms()
        if args.dataset == 'imagenet':
            dataset = torchvision.datasets.ImageNet(args.data_path, 'train', transform=transform)
        elif args.dataset == 'celeba':
            dataset = torchvision.datasets.CelebA(args.data_path, 'train', transform=transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        encodings = resnet_encode(model, dataloader, args)
    elif args.model == 'vit':
        transform = ViT_L_16_Weights.DEFAULT.transforms()
        if args.dataset == 'imagenet':
            dataset = torchvision.datasets.ImageNet(args.data_path, 'train', transform=transform)
        elif args.dataset == 'celeba':
            dataset = torchvision.datasets.CelebA(args.data_path, 'train', transform=transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        encodings = vit_encode(model, dataloader, args)
    
    np.save(args.encoding_file, encodings.numpy())


def resnet_encode(resnet, dataloader, args):
    """
        Perform encoding with resnet
    """
    encodings = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(f'cuda:{args.gpu}')
            encoding = resnet(x)
            encodings.append(encoding.cpu())

            if i % 100 == 0:
                print(f"Batch {i}/{len(dataloader)}")
    encodings = torch.cat(encodings, dim=0)
    return encodings
    
def vit_encode(vit, dataloader, args):
    """
        Perform encoding with ViT
    """
    encodings = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(f'cuda:{args.gpu}')
            feats = vit._process_input(x)
            batch_cls_token = vit.class_token.expand(x.shape[0], -1, -1)
            feats = torch.cat([batch_cls_token, feats], dim=1)
            feats = vit.encoder(feats)
            feats = feats[:, 0] # we only want the representation of cls-token
            encodings.append(feats.cpu())
            if i % 10 == 0:
                print(f"Batch {i}/{len(dataloader)}")
            del x
    encodings = torch.cat(encodings, dim=0)
    return encodings

def cluster_encodings(args):
    """
        Given a path to encodings, apply clustering.
    """
    encodings = np.load(args.encoding_file + '.npy')

    print(f"Clustering using {args.clustering}")

    if args.clustering == 'kmeans':
        mini_kmeans = MiniBatchKMeans(args.num_clusters, max_iter=500, batch_size=1024)
        clusters = mini_kmeans.fit_predict(encodings)

    elif args.clustering == 'torch-kmeans':
        encodings = torch.from_numpy(encodings)
        idx = torch.randint(low=0, high=len(encodings), size=(int(0.1*len(encodings)),))
        encodings = encodings[idx]
        cluster_ids, cluster_centers = kmeans(X=encodings, num_clusters=args.num_clusters,
                                                device=torch.device(f'cuda:{args.gpu}'))
        clusters = cluster_ids.cpu().numpy()
        cluster_centers = cluster_centers.cpu().numpy()
        np.save(f'{args.cluster_file}_centers', clusters)

    elif args.clustering == 'dbscan':
        dbscan = DBSCAN(args.dbscan_eps)
        clusters = dbscan.fit_predict(encodings)
    
    np.save(args.cluster_file, clusters)


def main(args):
    rt = rtpt.RTPT('JS', 'Deep_Image_Clustering', 1)
    rt.start()
    if args.reuse_encodings:
        print(f"Reusing encodings: {args.encoding_file}")
        cluster_encodings(args)
    else:
        print("Computing new encodings...")
        encode_imagenet(args)
        cluster_encodings(args)
    rt.step()


parser = argparse.ArgumentParser()

parser.add_argument('--reuse-encodings', action='store_true')
parser.add_argument('--encoding-file')
parser.add_argument('--cluster-file')
parser.add_argument('--clustering', default='kmeans')
parser.add_argument('--num-clusters', default=1000, type=int)
parser.add_argument('--dbscan-eps', default=0.5, type=float)
parser.add_argument('--model', default='resnet')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--data-path', default='/storage-01/datasets/imagenet')
parser.add_argument('--dataset', default='imagenet')

args = parser.parse_args()

main(args)
