import numpy as np
import os
import torch
import errno
from PIL import Image
from numproto import proto_to_ndarray

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
    
class ProtobufNumpyArray:
    """
        Class needed to deserialize numpy-arrays coming from flower
    """
    def __init__(self, bytes) -> None:
        self.ndarray = bytes

def flwr_params_to_numpy(params):
    meta_info = params.tensors[-1]
    adj = params.tensors[-2]
    parameter_bytes = params.tensors[:-2]
    pnpa_meta = ProtobufNumpyArray(meta_info)
    pnpa_adj = ProtobufNumpyArray(adj)

    meta_info = proto_to_ndarray(pnpa_meta)
    adj = proto_to_ndarray(pnpa_adj)
    
    parameters = []
    for p in parameter_bytes:
        param_bytes = ProtobufNumpyArray(p)
        parameters.append(proto_to_ndarray(param_bytes))

    return parameters, adj, meta_info

def get_data_by_cluster(clusters, cluster_n):
    data_idx = np.argwhere(clusters == cluster_n).flatten()
    return data_idx

def get_data_loader_mean(loader):
    mean = None
    for x, _ in loader:
        x = x.permute((0, 2, 3, 1))
        if mean is None:
            mean = torch.mean(x, 0)
        else:
            mean += torch.mean(x, 0)
    return mean / len(loader)