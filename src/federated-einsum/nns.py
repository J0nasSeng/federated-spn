import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import softmax_temp

class MLP(nn.Module):

    def __init__(self, in_dim, out_dims, h_dims) -> None:
        super().__init__()
        dims = [in_dim, *h_dims]
        out_dims_flat = [np.prod(o) for o in out_dims]
        self.out_dims = out_dims
        self.linear_layers = nn.ModuleList([nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))])
        self.heads = nn.ModuleList([nn.Linear(h_dims[-1], out_dims_flat[i]) for i in range(len(out_dims))])

    def forward(self, x_prev, y):
        x = [x_.squeeze().reshape(x_.shape[0], -1) for x_ in x_prev]
        x = torch.cat(x, dim=1)
        y_oh = F.one_hot(y, num_classes=1000)
        if len(y_oh.shape) < 2:
            y_oh = y_oh.unsqueeze(0)
        #x = torch.cat([x, y_oh], dim=1)
        for linear in self.linear_layers:
            x = linear(x)
            x = torch.relu(x)
        
        head_out = [h(x) for h in self.heads]
        out = []
        # normalize each head's output to yield valid einsum parameters
        for i, (o, d) in enumerate(zip(head_out, self.out_dims)):
            if i > 1:
                if len(d) == 4:
                    b = o.shape[0]
                    k1, k2, k3, r = d
                    out_o = o.reshape(b, k1*k2, k3, r)
                    out_o = torch.softmax(out_o, dim=1)
                    out_o = out_o.reshape(b, k1, k2, k3, r)
                    out.append(out_o)
                elif len(d) == 3:
                    out_o = torch.softmax(o, dim=1)
                    # bring back to old shape
                    out_o = torch.unsqueeze(out_o, dim=1)
                    out_o = torch.unsqueeze(out_o, dim=1)
                    out.append(out_o)
            else:
                b = o.shape[0]
                num_vars, k, a, t = d
                if i == 1:
                    o = torch.sigmoid(o) # avoid sigma being <= 0
                out_o = o.reshape(b, num_vars, k, a, t)
                out.append(out_o)
        
        leaf_params = torch.cat(out[:2], dim=-1)
        out = [leaf_params] + out[2:]
        return out
    
class CNN(nn.Module):

    def __init__(self, out_dims) -> None:
        super().__init__()
        out_dims_flat = [np.prod(o) for o in out_dims]
        self.out_dims = out_dims
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 128)
        self.heads = nn.ModuleList([nn.Linear(128, out_dims_flat[i]) for i in range(len(out_dims))])
    
    def _prepare_image(self, x):
        assert len(x) == 3, 'Received invalid number of patches, must be 3'
        h = x[0].shape[2]
        w = x[0].shape[3]
        b = x[0].shape[0]
        img = torch.zeros(b, 3, h*2, w*2).to(x[0].device)
        above, beside, diag = x
        img[:, :, 0:h, 0:w] = diag
        img[:, :, 0:h, w:(2*w)] = above
        img[:, :, h:(h*2), 0:w] = beside
        return img

    def forward(self, x, y):
        x = self._prepare_image(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        y_oh = F.one_hot(y, num_classes=1000)
        if len(y_oh.shape) < 2:
            y_oh = y_oh.unsqueeze(0)
        #x = torch.cat([x, y_oh], dim=1)
        x = F.relu(self.fc2(x))
        head_out = [h(x) for h in self.heads]
        out = []
        # normalize each head's output to yield valid einsum parameters
        for i, (o, d) in enumerate(zip(head_out, self.out_dims)):
            if i > 1:
                if len(d) == 4:
                    b = o.shape[0]
                    k1, k2, k3, r = d
                    out_o = o.reshape(b, k1*k2, k3, r)
                    out_o = torch.softmax(out_o, dim=1)
                    out_o = out_o.reshape(b, k1, k2, k3, r)
                    out.append(out_o)
                elif len(d) == 3:
                    out_o = torch.softmax(o, dim=1)
                    # bring back to old shape
                    out_o = torch.unsqueeze(out_o, dim=1)
                    out_o = torch.unsqueeze(out_o, dim=1)
                    out.append(out_o)
            else:
                b = o.shape[0]
                num_vars, k, a, t = d
                if i == 1:
                    o = torch.sigmoid(o) # avoid sigma being <= 0
                out_o = o.reshape(b, num_vars, k, a, t)
                out.append(out_o)
        
        leaf_params = torch.cat(out[:2], dim=-1)
        out = [leaf_params] + out[2:]
        return out
    
class CNNCondMixEin(nn.Module):
    def __init__(self, num_clusters) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_clusters)
        self.bn = nn.BatchNorm1d(num_clusters)
    
    def _prepare_image(self, x):
        assert len(x) == 3, 'Received invalid number of patches, must be 3'
        h = x[0].shape[2]
        w = x[0].shape[3]
        b = x[0].shape[0]
        img = torch.zeros(b, 3, h*2, w*2).to(x[0].device)
        above, beside, diag = x
        img[:, :, 0:h, 0:w] = diag
        img[:, :, 0:h, w:(2*w)] = above
        img[:, :, h:(h*2), 0:w] = beside
        return img

    def forward(self, x, y):
        x = self._prepare_image(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #y_oh = F.one_hot(y, num_classes=1000)
        #if len(y_oh.shape) < 2:
        #    y_oh = y_oh.unsqueeze(0)
        #x = torch.cat([x, y_oh], dim=1)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = self.bn(x)
        out = softmax_temp(x, t=1.)
        return out