import torch
import pyjuice as juice
import pyjuice.nodes.distributions as juice_dists
from pyjuice.structures import PD, RAT_SPN
from torchvision.datasets import ImageNet, SVHN, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from rtpt import RTPT
import numpy as np

batch_size = 256


def rgb_to_ycocg(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to YCoCg color space.
    Args:
        rgb (torch.Tensor): Input tensor of shape (N, C, H, W) with RGB channels.
    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W) with YCoCg channels.
    """
    # Ensure input is in the expected range [0, 1] or [0, 255]
    if rgb.max() > 1:
        rgb = rgb / 255.0

    # Conversion matrix for RGB to YCoCg
    transformation_matrix = torch.tensor(
        [[0.25, 0.5, 0.25], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]],  # Y  # Co  # Cg
        dtype=rgb.dtype,
        device=rgb.device,
    )

    # Reshape RGB channels to apply the matrix
    rgb = rgb.permute(0, 2, 3, 1)  # Change to (N, H, W, C)
    ycocg = torch.matmul(
        rgb, transformation_matrix.T
    )  # torch.einsum('nhwc,cc->nhwc', rgb, transformation_matrix)
    ycocg = ycocg.permute(0, 3, 1, 2)  # Change back to (N, C, H, W)
    # make all dimensions between 0 and 1
    ycocg[:, [1, 2], :, :] += 0.5

    return ycocg


def bits_per_dim(nll, num_features):
    # If NLL is summed across a batch, take the mean
    nll_mean = nll.mean() if isinstance(nll, torch.Tensor) else np.mean(nll)

    # Compute bits per dimension
    bpd = nll_mean / (num_features * np.log(2))
    return bpd


def load_dataset(ds_name, split="train"):
    if ds_name == "imagenet":
        transform = Compose([ToTensor(), Resize((64, 64))])
        dataset = ImageNet(
            "/storage-01/datasets/imagenet/", transform=transform, split=split
        )
        data_shape = (64, 64, 3)
    elif ds_name == "imagenet32":
        transform = Compose([ToTensor(), Resize((32, 32))])
        dataset = ImageNet(
            "/storage-01/datasets/imagenet/", transform=transform, split=split
        )
        data_shape = (32, 32, 3)
    elif ds_name == "celeba":
        transform = Compose([ToTensor(), Resize((32, 32))])
        dataset = CelebA("/storage-01/datasets/", transform=transform, split=split)
        data_shape = (32, 32, 3)
    return dataset, data_shape


def train(ds_name, num_epochs):

    rt = RTPT("JS", "PyJuice", num_epochs)
    rt.start()
    device = torch.device(f"cuda:{4}")
    torch.manual_seed(2)
    dataset, shape = load_dataset(ds_name)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    input_dist = juice_dists.Categorical(256)
    arch = PD(shape, 256, input_dist=input_dist, split_intervals=4)
    print(arch)
    model = juice.compile(arch)
    torch.cuda.set_device(device)
    model = model.to(device)
    transformation_matrix = torch.tensor(
        [[0.25, 0.5, 0.25], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]],  # Y  # Co  # Cg
        device=device,
    )

    for e in range(num_epochs):

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
            x = x * 255
            x = x.reshape(x.shape[0], -1).long()

            # This is equivalent to zeroing out the parameter gradients of a neural network
            model.init_param_flows(flows_memory=0.0)
            # Forward pass
            lls = model(x)
            # Backward pass
            lls.mean().backward()
            total_ll += lls.sum().detach().cpu() / (len(loader) * loader.batch_size)
            # Mini-batch EM
            model.mini_batch_em(step_size=0.02, pseudocount=0.001)

            if i % 20 == 0:
                # print(lls.flatten())
                print(
                    f"Epoch {e+1}/{num_epochs}: \t Iter: {i}/{len(loader)}: \t LL: {total_ll}"
                )

        print(f"Epoch {e+1}/{num_epochs} \t LL: {total_ll}")
        rt.step()

    return model, device


def evaluate(model, dataset, device):

    loader = DataLoader(dataset, batch_size=256, num_workers=0)

    total_ll = 0.0

    torch.cuda.set_device(device)
    model = model.to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)
            x = x.to(device)

            lls = model(x)
            total_ll += lls.sum()

    return total_ll / (len(loader) * loader.batch_size)


model, device = train("celeba", 10)
test_set, shape = load_dataset("celeba", "valid")
ll = evaluate(model, test_set, device)
nats = ll.numpy() / np.prod(shape)
print(f"LL: {ll}")
print(f"NATS: {nats}")
