import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod


class SPNNode(nn.Module, ABC):
    """
    Abstract base class for all SPN nodes.
    Replaces components from spn.structure.Base
    """

    def __init__(self):
        super().__init__()
        self.scope = []
        self.id = None
        self.num_dist = 1  # Number of distributions for vectorized operations

    @abstractmethod
    def forward(self, x):
        pass


class Sum(SPNNode):
    """
    PyTorch-based Sum node implementation using einsum operations.
    Replaces spn.structure.Base.Sum
    """

    def __init__(
        self,
        weights: Optional[Union[List[float], torch.Tensor]] = None,
        children: Optional[List[SPNNode]] = None,
    ):
        super().__init__()
        self.children = children or []

        # Initialize weights
        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(weights)
        else:
            # Initialize with uniform weights
            num_children = len(self.children) if self.children else 1
            self.weights = nn.Parameter(torch.ones(num_children) / num_children)

        # Ensure scope is union of children scopes
        if self.children:
            self.scope = list(set().union(*[child.scope for child in self.children]))

    def forward(self, x):
        """
        Forward pass using einsum for efficient computation.
        x: (batch_size, num_features) or (batch_size, num_dist, num_features)
        """
        if not self.children:
            # Leaf case - should not happen for Sum nodes
            raise ValueError("Sum node must have children")

        # Compute log probabilities from children
        child_log_probs = []
        for child in self.children:
            child_log_prob = child(x)
            child_log_probs.append(child_log_prob)

        # Stack child log probabilities: (num_children, batch_size, num_dist)
        child_log_probs = torch.stack(child_log_probs, dim=0)

        # Log-sum-exp trick for numerical stability
        # Add log weights: (num_children, 1, 1) + (num_children, batch_size, num_dist)
        log_weights = torch.log(self.weights).view(-1, 1, 1)
        weighted_log_probs = log_weights + child_log_probs

        # Use einsum for weighted sum in log domain
        # LogSumExp over children dimension
        max_val = torch.max(weighted_log_probs, dim=0, keepdim=True)[0]
        exp_probs = torch.exp(weighted_log_probs - max_val)
        sum_exp = torch.sum(exp_probs, dim=0)
        result = torch.log(sum_exp) + max_val.squeeze(0)

        return result


class Product(SPNNode):
    """
    PyTorch-based Product node implementation using einsum operations.
    Replaces spn.structure.Base.Product
    """

    def __init__(self, children: Optional[List[SPNNode]] = None):
        super().__init__()
        self.children = children or []

        # Ensure scope is union of children scopes (should be disjoint for valid SPNs)
        if self.children:
            self.scope = list(set().union(*[child.scope for child in self.children]))

    def forward(self, x):
        """
        Forward pass using einsum for efficient computation.
        """
        if not self.children:
            # Leaf case - should not happen for Product nodes
            raise ValueError("Product node must have children")

        # Compute log probabilities from children and sum them (log domain multiplication)
        total_log_prob = None
        for child in self.children:
            child_log_prob = child(x)
            if total_log_prob is None:
                total_log_prob = child_log_prob
            else:
                total_log_prob = total_log_prob + child_log_prob

        return total_log_prob


class Leaf(SPNNode):
    """
    Base class for leaf nodes in SPN.
    """

    def __init__(self, scope_var: int):
        super().__init__()
        self.scope = [scope_var]
        self.scope_var = scope_var

    @abstractmethod
    def forward(self, x):
        pass


class GaussianLeaf(Leaf):
    """
    Gaussian leaf node implementation.
    """

    def __init__(self, scope_var: int, mean: float = 0.0, var: float = 1.0):
        super().__init__(scope_var)
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.log_var = nn.Parameter(torch.log(torch.tensor(var, dtype=torch.float32)))

    def forward(self, x):
        """
        Compute log probability of Gaussian distribution.
        """
        if x.dim() == 2:
            # x: (batch_size, num_features)
            x_var = x[:, self.scope_var]
        else:
            # x: (batch_size, num_dist, num_features)
            x_var = x[:, :, self.scope_var]

        var = torch.exp(self.log_var)

        # Gaussian log probability
        log_prob = (
            -0.5 * torch.log(2 * np.pi * var) - 0.5 * (x_var - self.mean) ** 2 / var
        )

        if x.dim() == 2:
            log_prob = log_prob.unsqueeze(1)  # Add num_dist dimension

        return log_prob


class CategoricalLeaf(Leaf):
    """
    Categorical leaf node implementation.
    """

    def __init__(self, scope_var: int, num_categories: int):
        super().__init__(scope_var)
        self.num_categories = num_categories
        # Use log probabilities for numerical stability
        self.log_probs = nn.Parameter(torch.randn(num_categories))

    def forward(self, x):
        """
        Compute log probability of Categorical distribution.
        """
        if x.dim() == 2:
            x_var = x[:, self.scope_var].long()
        else:
            x_var = x[:, :, self.scope_var].long()

        # Normalize log probabilities
        log_probs_normalized = F.log_softmax(self.log_probs, dim=0)

        # Select probabilities based on observed values
        if x.dim() == 2:
            log_prob = log_probs_normalized[x_var].unsqueeze(1)
        else:
            log_prob = log_probs_normalized[x_var]

        return log_prob
