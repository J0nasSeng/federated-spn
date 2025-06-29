import torch
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional
import numpy as np
from torch_spn.base import SPNNode, Sum


def log_likelihood(spn: SPNNode, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Compute log-likelihood of data under the SPN.
    Replaces spn.algorithms.Inference.log_likelihood

    Args:
        spn: Root SPN node
        data: Input data of shape (batch_size, num_features) or numpy array

    Returns:
        Log-likelihood values of shape (batch_size,)
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # Handle NaN values (for marginal inference)
    mask = ~torch.isnan(data)

    # Forward pass through SPN
    with torch.no_grad():
        log_probs = spn(data)

    # If output has num_dist dimension, take mean or sum
    if log_probs.dim() > 1:
        log_probs = log_probs.squeeze()

    return log_probs


def mpe(spn: SPNNode, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Most Probable Explanation (MPE) inference.
    Replaces spn.algorithms.MPE.mpe

    Args:
        spn: Root SPN node
        data: Input data with some variables set to NaN for inference

    Returns:
        MPE assignments for missing variables
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # For now, implement a simple version
    # In practice, this would require modifying the forward pass to use max instead of sum
    # and implementing backtracking

    # Create a copy for modification
    result = data.clone()
    nan_mask = torch.isnan(data)

    if not torch.any(nan_mask):
        return result

    # Simple approach: try different values and pick the one with highest likelihood
    # This is a placeholder - proper MPE requires specialized forward pass
    best_ll = float("-inf")
    best_assignment = result.clone()

    # For categorical variables (assume integer values 0-9 for simplicity)
    for var_idx in range(data.shape[1]):
        if torch.any(nan_mask[:, var_idx]):
            for val in range(10):  # Try values 0-9
                temp_result = result.clone()
                temp_result[nan_mask[:, var_idx], var_idx] = val

                ll = log_likelihood(spn, temp_result)
                if torch.sum(ll) > best_ll:
                    best_ll = torch.sum(ll)
                    best_assignment = temp_result.clone()

    return best_assignment


def marginal_inference(
    spn: SPNNode, data: Union[torch.Tensor, np.ndarray], query_vars: List[int]
) -> torch.Tensor:
    """
    Marginal inference over specified variables.

    Args:
        spn: Root SPN node
        data: Input data
        query_vars: Variables to marginalize over

    Returns:
        Marginal probabilities
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # Set query variables to NaN to marginalize
    marginalized_data = data.clone()
    marginalized_data[:, query_vars] = float("nan")

    return log_likelihood(spn, marginalized_data)


class EMOptimizer:
    """
    EM optimization for SPNs using PyTorch.
    Replaces spn.algorithms.EM.EM_optimization
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self, spn: SPNNode, data: Union[torch.Tensor, np.ndarray]
    ) -> List[float]:
        """
        Perform EM optimization on the SPN.

        Args:
            spn: Root SPN node
            data: Training data

        Returns:
            List of log-likelihood values during training
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        optimizer = torch.optim.Adam(spn.parameters(), lr=self.learning_rate)
        ll_history = []

        prev_ll = float("-inf")

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Forward pass
            log_probs = spn(data)

            # Negative log-likelihood loss
            loss = -torch.mean(log_probs)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Project parameters to valid ranges (e.g., sum weights to simplex)
            self._project_parameters(spn)

            current_ll = -loss.item()
            ll_history.append(current_ll)

            # Check convergence
            if abs(current_ll - prev_ll) < self.tolerance:
                print(f"EM converged after {iteration + 1} iterations")
                break

            prev_ll = current_ll

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, LL: {current_ll:.6f}")

        return ll_history

    def _project_parameters(self, node: SPNNode):
        """Project parameters to valid ranges."""
        if isinstance(node, Sum):
            # Project weights to simplex
            with torch.no_grad():
                node.weights.data = F.softmax(node.weights.data, dim=0)

        # Recursively project children
        if hasattr(node, "children"):
            for child in node.children:
                self._project_parameters(child)


def EM_optimization(
    spn: SPNNode,
    data: Union[torch.Tensor, np.ndarray],
    learning_rate: float = 0.01,
    max_iterations: int = 100,
) -> List[float]:
    """
    Convenience function for EM optimization.
    Replaces spn.algorithms.EM.EM_optimization
    """
    optimizer = EMOptimizer(learning_rate, max_iterations)
    return optimizer.optimize(spn, data)


def get_nodes_by_type(spn: SPNNode, node_type: type) -> List[SPNNode]:
    """
    Get all nodes of a specific type from the SPN.
    Replaces spn.structure.Base.get_nodes_by_type
    """
    nodes = []

    def traverse(node):
        if isinstance(node, node_type):
            nodes.append(node)

        if hasattr(node, "children"):
            for child in node.children:
                traverse(child)

    traverse(spn)
    return nodes
