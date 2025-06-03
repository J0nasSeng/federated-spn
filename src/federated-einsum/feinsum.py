import numpy as np
import torch.nn.grad
from torchvision.datasets import ImageNet, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader, TensorDataset
from einsum import EinsumNetwork, Graph
import config
import torch
import os
import logging
import sys
from multiprocessing import Process
from rtpt import RTPT
import pandas as pd
from pathlib import Path

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)


def init_spn(device, num_vars, num_dims, use_em=True, num_classes=1):
    """
    Build a SPN (implemented as an einsum network). The structure is either
    the same as proposed in https://arxiv.org/pdf/1202.3732.pdf (referred to as
    poon-domingos) or a binary tree.

    In case of poon-domingos the image is split into smaller hypercubes (i.e. a set of
    neighbored pixels) where each pixel is a random variable. These hypercubes are split further
    until we operate on pixel-level. The spplitting is done randomly. For more information
    refer to the link above.
    """

    if config.structure == "poon-domingos":
        pd_delta = [[config.height / d, config.width / d] for d in config.pd_num_pieces]
        graph = Graph.poon_domingos_structure(
            shape=(config.height, config.width), delta=pd_delta
        )
    elif config.structure == "binary-trees":
        graph = Graph.random_binary_trees(
            num_var=config.num_vars,
            depth=config.depth,
            num_repetitions=config.num_repetitions,
        )
    elif config.structure == "flat-binary-tree":
        graph = Graph.binary_tree_spn(shape=(config.height, config.width))
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
        num_var=num_vars,
        num_dims=num_dims,
        num_classes=num_classes,
        num_sums=config.K,
        num_input_distributions=config.K,
        exponential_family=config.exponential_family,
        exponential_family_args=config.exponential_family_args,
        online_em_frequency=config.online_em_frequency,
        online_em_stepsize=config.online_em_stepsize,
        use_em=use_em,
    )

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    print(sum([p.numel() for p in einet.parameters()]))
    return einet


def train_classifier(
    img_ids, num_epochs, device_id, chk_path, cluster_count, dataset="imagenet"
):
    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    if not os.path.exists(chk_path):
        path = Path(chk_path)
        path.mkdir(parents=True)
    logging.info("Starting Training...")
    log_likelihoods = []
    device = torch.device(f"cuda:{device_id}")
    if dataset == "imagenet":
        transform = Compose([ToTensor(), Resize(112, antialias=True), CenterCrop(112)])
        ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform)
        num_vars = 112 * 112
        num_dims = 3
    elif dataset == "imagenet32":
        transform = Compose([ToTensor(), Resize(32, antialias=True), CenterCrop(32)])
        ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform)
        num_vars = 32 * 32
        num_dims = 3
    elif dataset == "celeba":
        transform = Compose([ToTensor(), Resize(64, antialias=True), CenterCrop(64)])
        ds = CelebA("/storage-01/datasets/", transform=transform)
        num_vars = 64 * 64
        num_dims = 3
    subset = Subset(ds, img_ids)
    loader = DataLoader(subset, batch_size=config.batch_size, num_workers=2)
    einet = init_spn(device, num_vars, num_dims, num_classes=2)
    optimizer = torch.optim.Adam(einet.parameters(), lr=0.01)
    cross_entropy = torch.nn.CrossEntropyLoss()
    for epoch_count in range(num_epochs):
        einet.train()

        total_ll = 0.0
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(device)

            # make y one-hot encoded because model output is 2d
            y = y.to(device)[:, 0].to(torch.float32)
            y_scd = torch.zeros(y.shape[0])
            y_scd = 1 - y
            y = torch.stack((y, y_scd), dim=1)

            x = x.permute((0, 2, 3, 1))
            x = x.reshape(x.shape[0], num_vars, num_dims)
            lls = einet.forward(x)
            # lls = torch.softmax(lls, dim=1)

            # ll_sample = EinsumNetwork.log_likelihoods(outputs)
            loss = cross_entropy(lls, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(einet.parameters(), 1.0)

            optimizer.step()

            total_ll += loss.item()

            # if i % 10 == 0:
            #    logging.info('Epoch {:03d} \t Step {:03d} \t LL {:03f}'.format(epoch_count, i, total_ll))
        total_ll = total_ll / (len(loader) * loader.batch_size)
        log_likelihoods.append(total_ll)
        logging.info("Epoch {:03d} \t LL={:03f}".format(epoch_count, total_ll))

    transform = Compose([ToTensor(), Resize(64, antialias=True), CenterCrop(64)])
    ds = CelebA("/storage-01/datasets/", transform=transform, split="valid")
    loader = DataLoader(ds, batch_size=config.batch_size, num_workers=2)

    with torch.no_grad():
        accs = []
        for x, y in loader:
            x = x.to(device)

            # make y one-hot encoded because model output is 2d
            y = y.to(device)[:, 0].to(torch.float32)
            y_scd = torch.zeros(y.shape[0])
            y_scd = 1 - y
            y = torch.stack((y, y_scd), dim=1)

            x = x.permute((0, 2, 3, 1))
            x = x.reshape(x.shape[0], num_vars, num_dims)
            lls = einet.forward(x)

            _, y_pred = torch.max(lls, dim=1)
            _, y = torch.max(y, dim=1)
            num_correct = (y_pred == y).sum().item()
            accs.append(num_correct / y_pred.shape[0])

    print(np.mean(accs))

    # torch.save(einet, os.path.join(chk_path, f'chk_{cluster_count}.pt'))
    df = pd.DataFrame(data=log_likelihoods, columns=["lls"])
    df.to_csv(os.path.join(chk_path, f"chk_{cluster_count}.csv"))
    return einet


def train(img_ids, num_epochs, device_id, chk_path, cluster_count, dataset="imagenet"):
    """
    Training loop to train the SPN. Follows EM-procedure.
    """
    if not os.path.exists(chk_path):
        path = Path(chk_path)
        path.mkdir(parents=True)
    logging.info("Starting Training...")
    log_likelihoods = []
    device = torch.device(f"cuda:{device_id}")
    num_vars = config.num_vars
    num_dims = config.num_dims
    if dataset == "imagenet":
        transform = Compose([ToTensor(), Resize((config.height, config.width))])
        ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform)
    elif dataset == "imagenet32":
        transform = Compose([ToTensor(), Resize((config.height, config.width))])
        ds = ImageNet("/storage-01/datasets/imagenet/", transform=transform)
    elif dataset == "celeba":
        transform = Compose([ToTensor(), Resize((config.height, config.width))])
        ds = CelebA("/storage-01/datasets/", transform=transform)

    subset = Subset(ds, img_ids)
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
    )
    einet = init_spn(device, num_vars, num_dims)
    transformation_matrix = torch.tensor(
        [[0.25, 0.5, 0.25], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]],  # Y  # Co  # Cg
        device=device,
    )
    for epoch_count in range(num_epochs):
        einet.train()

        total_ll = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            # Reshape RGB channels to apply the matrix
            x = x.permute(0, 2, 3, 1)  # Change to (N, H, W, C)
            ycocg = torch.matmul(x, transformation_matrix.T)
            ycocg = ycocg.permute(0, 3, 1, 2)  # Change back to (N, C, H, W)
            # make all dimensions between 0 and 1
            ycocg[:, [1, 2], :, :] += 0.5
            x = ycocg
            # x = x * 255
            x = x.reshape(x.shape[0], num_vars, num_dims)
            ll_sample = einet.forward(x)
            # ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().cpu() / (
                len(loader) * loader.batch_size
            )

            if i % 20 == 0:
                logging.info(
                    "Epoch {:03d} \t Step {:03d} \t LL {:03f}".format(
                        epoch_count, i, total_ll
                    )
                )
        log_likelihoods.append(total_ll)
        logging.info("Epoch {:03d} \t LL={:03f}".format(epoch_count, total_ll))

        einet.em_update()
    torch.save(einet, os.path.join(chk_path, f"chk_{cluster_count}.pt"))
    df = pd.DataFrame(data=log_likelihoods, columns=["lls"])
    df.to_csv(os.path.join(chk_path, f"chk_{cluster_count}.csv"))
    return einet


def train_mixture(clusters, dataset="imagenet", task="density_estimation"):
    unique_clusters = np.unique(clusters)
    num_slices = int(np.ceil(len(unique_clusters) / config.num_processes))
    batched_unique_clusters = np.array_split(unique_clusters, num_slices)
    rt = RTPT("JS", "FedEinsum", len(unique_clusters))
    rt.start()
    for cluster_batch in batched_unique_clusters:
        processes = []
        for i, rc in enumerate(cluster_batch):
            idx = i % len(config.devices)
            device_id = config.devices[idx]
            # img_ids = np.argwhere(clusters == rc).flatten()
            img_ids = np.random.randint(
                0, len(clusters), size=int(len(clusters) / len(unique_clusters))
            ).flatten()

            print(f"Cluster-size={len(img_ids)}")
            checkpoint_dir = "./checkpoints/imagenet/v5/checkpoints_ceinet_2clusters/"
            if task == "density_estimation":
                p = Process(
                    target=train,
                    args=(
                        img_ids,
                        config.num_epochs,
                        device_id,
                        checkpoint_dir,
                        rc,
                        dataset,
                    ),
                )
            else:
                p = Process(
                    target=train_classifier,
                    args=(
                        img_ids,
                        config.num_epochs,
                        device_id,
                        checkpoint_dir,
                        rc,
                        dataset,
                    ),
                )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        rt.step()


RAND_CLUSTERS = False

# clusters = np.load('/storage-01/ml-jseng/imagenet-clusters/vit_cluster_minibatch_16.npy')
clusters = np.load("/storage-01/ml-jseng/imagenet-clusters/vit_clusters_2_centers.npy")
print(clusters.shape)

if RAND_CLUSTERS:
    # If we shuffle cluster assignments randomly, this is the same as distributing the images randomly.
    clusters = np.random.permutation(clusters)

# encodings = np.load('/storage-01/ml-jseng/imagenet-clusters/vit_enc.npy')2
# train einets in parallel. Start num_slices processes in parallel, wait
# until they finished and start next batch
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    train_mixture(clusters, "imagenet", task="density_estimation")

# weights = np.array(cluster_sizes) / np.sum(cluster_sizes)
# cluster_idx = np.random.choice(np.arange(len(weights)), 3, p=weights)
# for cidx in cluster_idx:
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
# rand_clusters = [2400] #np.random.randint(0, clusters.max(), size=50)
# for rc in np.unique(rand_clusters):
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
# weights = np.array(cluster_sizes) / np.sum(cluster_sizes)
# mixture = EinetMixture.EinetMixture(weights, root_einets)
# root_einets.append(mixture)
#
# samples = mixture.sample(25)
# samples = samples.reshape(-1, config.height, config.width, config.num_dims)
# img_path = os.path.join('./', f'samples.png')
# save_image_stack(samples, 5, 5, img_path, margin_gray_val=0., frame=2, frame_gray_val=0.0)
