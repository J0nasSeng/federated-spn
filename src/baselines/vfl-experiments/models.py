import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNetEncoder, EmbeddingGenerator
from pytorch_tabnet.utils import create_group_matrix
from vit_pytorch.simple_vit import Transformer, posemb_sincos_2d, pair, Rearrange
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        num_clients,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.is_multi_task = isinstance(output_dim, list)
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        
        self.encoder = TabNetEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                n_independent=n_independent,
                n_shared=n_shared,
                epsilon=epsilon,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
                mask_type=mask_type,
                group_attention_matrix=group_attention_matrix
            )
        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return res, M_loss
        #if self.is_multi_task:
        #    # Result will be in list format
        #    out = []
        #    for task_mapping in self.multi_task_mappings:
        #        out.append(task_mapping(res))
        #else:
        #    out = self.final_mapping(res)
        #return out

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet(torch.nn.Module):
    def __init__(
        self,
        num_clients,
        input_dims,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        grouped_features=[],
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.is_multi_task = isinstance(output_dim, list)

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedders = nn.ModuleList()
        self.tabnets = nn.ModuleList()
        for i in range(num_clients):
            group_attention_matrix = create_group_matrix(grouped_features[i], input_dims[i])
            embedder = EmbeddingGenerator(input_dims[i],
                                           cat_dims[i],
                                           cat_idxs[i],
                                           cat_emb_dim[i],
                                           group_attention_matrix)
            self.embedders.append(embedder)
            self.post_embed_dim = embedder.post_embed_dim

            tabnet = TabNetNoEmbeddings(
                num_clients,
                self.post_embed_dim,
                output_dim,
                n_d,
                n_a,
                n_steps,
                gamma,
                n_independent,
                n_shared,
                epsilon,
                virtual_batch_size,
                momentum,
                mask_type,
                group_attention_matrix
            )
            self.tabnets.append(tabnet)
        self.final_mapping = nn.Sequential(
            Linear(num_clients * n_d, int((num_clients * n_d) / 2)),
            nn.ReLU(),
            nn.Linear(int((num_clients * n_d) / 2), output_dim)
        )
        #initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        x_tabnet =  []
        out = torch.zeros(x[0].shape[0], self.output_dim)
        M_l = 0.0
        for i, x_ in enumerate(x):
            x_ = self.embedders[i](x_)
            x_, M_loss = self.tabnets[i](x_)
            M_l += M_loss / len(x)
            x_tabnet.append(x_)
            #out += x_ / len(x)
        #return out
        x_ = torch.column_stack(x_tabnet)
        return self.final_mapping(x_), M_l

    def forward_masks(self, x):
        x_tabnet =  []
        for i, x_ in enumerate(x):
            x_ = self.embedders[i](x_)
            x_ = self.tabnets[i].forward_masks(x_)
            x_tabnet.append(x_)
        x_ = torch.column_stack(x_tabnet)
        return x_
    

class SimpleViT(nn.Module):
    def __init__(self, *, num_clients, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embeddings = []
        self.transformers = []
        self.pos_embeddings = []
        self.to_latents = []

        for _ in range(num_classes):

            to_patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

            pos_embedding = posemb_sincos_2d(
                h = image_height // patch_height,
                w = image_width // patch_width,
                dim = dim,
            ) 

            transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

            self.pool = "mean"
            to_latent = nn.Identity()

            self.to_patch_embeddings.append(to_patch_embedding)
            self.transformers.append(transformer)
            self.pos_embeddings.append(pos_embedding)
            self.to_latents.append(to_latent)

        self.linear_head = nn.Linear(dim*num_clients, num_classes)

    def forward_client_transformers(self, imgs):
        out = []
        for i, img in enumerate(imgs):
            to_patch_embedding = self.to_patch_embeddings[i]
            pos_embedding = self.pos_embeddings[i]
            transformer = self.transformers[i]
            to_latent = self.to_latents[i]
            device = img.device

            x = to_patch_embedding(img)
            x += pos_embedding.to(device, dtype=x.dtype)

            x = transformer(x)
            x = x.mean(dim = 1)

            x = to_latent(x)

            out.append(x)
        return out

    def forward(self, imgs):
        client_outs = self.forward_client_transformers(imgs)
        client_outs = torch.stack(client_outs, dim=1)
        return self.linear_head(client_outs)