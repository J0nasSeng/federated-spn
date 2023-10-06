import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):

    def __init__(self, in_dim, out_dims, h_dims) -> None:
        super().__init__()
        dims = [in_dim, *h_dims]
        out_dims_flat = [np.prod(o) for o in out_dims]
        self.out_dims = out_dims
        self.linear_layers = nn.ModuleList([nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))])
        self.heads = nn.ModuleList([nn.Linear(h_dims[-1], out_dims_flat[i]) for i in range(len(out_dims))])

    def forward(self, x):
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
                    out_o = o.squeeze()
                    out_o = torch.softmax(out_o, dim=1)
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
    