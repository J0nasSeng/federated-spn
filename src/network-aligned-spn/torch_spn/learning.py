import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from torch_spn.base import CategoricalLeaf, GaussianLeaf, Product, SPNNode
from torch_spn.inference import EM_optimization


class Context:
    """
    Context class for SPN learning.
    Replaces spn.structure.Base.Context
    """

    def __init__(
        self,
        meta_types: Optional[List[Any]] = None,
        parametric_types: Optional[List[Any]] = None,
    ):
        self.meta_types = meta_types or []
        self.parametric_types = parametric_types or []
        self.domains = {}
        self.feature_types = []

    def add_domains(self, data: Union[np.ndarray, torch.Tensor]):
        """Add domain information from data."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        for i in range(data.shape[1]):
            column = data[:, i]
            # Remove NaN values for domain estimation
            valid_values = column[~np.isnan(column)]

            if len(valid_values) == 0:
                continue

            # Determine if categorical or continuous
            unique_values = np.unique(valid_values)

            if len(unique_values) <= 10 and np.all(
                np.equal(np.mod(valid_values, 1), 0)
            ):
                # Categorical
                self.domains[i] = {
                    "type": "categorical",
                    "values": unique_values.tolist(),
                    "num_categories": len(unique_values),
                }
                self.feature_types.append("categorical")
            else:
                # Continuous
                self.domains[i] = {
                    "type": "continuous",
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                }
                self.feature_types.append("continuous")


def learn_mspn(
    data: Union[np.ndarray, torch.Tensor],
    context: Context,
    min_instances_slice: int = 200,
    threshold: float = 0.3,
    depth: int = 3,
    num_clusters: int = 2,
) -> SPNNode:
    """
    Learn a Mixture of Sum-Product Networks.
    Replaces spn.algorithms.LearningWrappers.learn_mspn

    Args:
        data: Training data
        context: Context with domain information
        min_instances_slice: Minimum instances for splitting
        threshold: Threshold for independence test
        depth: Maximum depth of the SPN
        num_clusters: Number of clusters for mixture

    Returns:
        Root SPN node
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Build SPN recursively
    return _build_spn_recursive(
        data,
        context,
        list(range(data.shape[1])),
        min_instances_slice,
        threshold,
        depth,
        0,
    )


def _build_spn_recursive(
    data: np.ndarray,
    context: Context,
    scope: List[int],
    min_instances_slice: int,
    threshold: float,
    max_depth: int,
    current_depth: int,
) -> SPNNode:
    """
    Recursively build SPN structure.
    """

    # Base case: small data or max depth reached
    if len(data) < min_instances_slice or current_depth >= max_depth or len(scope) == 1:
        return _create_leaf_nodes(data, context, scope)

    # Try to split on variables (create Product node)
    if len(scope) > 1:
        # Simple variable splitting - can be made more sophisticated
        mid = len(scope) // 2
        left_scope = scope[:mid]
        right_scope = scope[mid:]

        # Check independence (simplified)
        if _test_independence(data, left_scope, right_scope, threshold):
            # Create Product node
            left_child = _build_spn_recursive(
                data,
                context,
                left_scope,
                min_instances_slice,
                threshold,
                max_depth,
                current_depth + 1,
            )
            right_child = _build_spn_recursive(
                data,
                context,
                right_scope,
                min_instances_slice,
                threshold,
                max_depth,
                current_depth + 1,
            )

            product_node = Product([left_child, right_child])
            product_node.scope = scope
            return product_node

    # Try to split on instances (create Sum node)
    if len(data) >= 2 * min_instances_slice:
        # Cluster instances
        num_clusters = min(3, len(data) // min_instances_slice)
        if num_clusters > 1:
            # Use KMeans for clustering
            if len(scope) > 1:
                cluster_data = data[:, scope]
            else:
                cluster_data = data[:, scope].reshape(-1, 1)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_data)

            children = []
            weights = []

            for cluster_id in range(num_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data_subset = data[cluster_mask]

                if len(cluster_data_subset) >= min_instances_slice // 2:
                    child = _build_spn_recursive(
                        cluster_data_subset,
                        context,
                        scope,
                        min_instances_slice,
                        threshold,
                        max_depth,
                        current_depth + 1,
                    )
                    children.append(child)
                    weights.append(len(cluster_data_subset) / len(data))

            if len(children) > 1:
                sum_node = Sum(weights, children)
                sum_node.scope = scope
                return sum_node

    # Fallback: create leaf nodes
    return _create_leaf_nodes(data, context, scope)


def _test_independence(
    data: np.ndarray, left_scope: List[int], right_scope: List[int], threshold: float
) -> bool:
    """
    Simple independence test between variable sets.
    """
    # For simplicity, assume independence if scopes are disjoint
    # In practice, you'd use statistical tests
    return len(set(left_scope) & set(right_scope)) == 0


def _create_leaf_nodes(data: np.ndarray, context: Context, scope: List[int]) -> SPNNode:
    """
    Create leaf nodes for the given scope.
    """
    if len(scope) == 1:
        # Single variable leaf
        var_idx = scope[0]

        if var_idx in context.domains:
            domain_info = context.domains[var_idx]

            if domain_info["type"] == "categorical":
                return CategoricalLeaf(var_idx, domain_info["num_categories"])
            else:
                # Continuous - use Gaussian
                mean = domain_info["mean"]
                var = domain_info["std"] ** 2
                return GaussianLeaf(var_idx, mean, var)
        else:
            # Default to Gaussian
            column_data = data[:, var_idx]
            valid_data = column_data[~np.isnan(column_data)]
            if len(valid_data) > 0:
                mean = float(np.mean(valid_data))
                var = float(np.var(valid_data)) + 1e-6  # Add small epsilon
            else:
                mean, var = 0.0, 1.0
            return GaussianLeaf(var_idx, mean, var)

    else:
        # Multiple variables - create Product of leaves
        children = []
        for var_idx in scope:
            child = _create_leaf_nodes(data, context, [var_idx])
            children.append(child)

        product_node = Product(children)
        product_node.scope = scope
        return product_node


class SPNLearner:
    """
    Advanced SPN learning with more sophisticated algorithms.
    """

    def __init__(self, min_instances_slice: int = 200, threshold: float = 0.3):
        self.min_instances_slice = min_instances_slice
        self.threshold = threshold

    def learn_structure_and_parameters(
        self, data: Union[np.ndarray, torch.Tensor], context: Optional[Context] = None
    ) -> SPNNode:
        """
        Learn both structure and parameters of SPN.
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if context is None:
            context = Context()
            context.add_domains(data)

        # Learn structure
        spn = learn_mspn(data, context, self.min_instances_slice, self.threshold)

        # Optimize parameters
        data_tensor = torch.tensor(data, dtype=torch.float32)
        EM_optimization(spn, data_tensor)

        return spn
