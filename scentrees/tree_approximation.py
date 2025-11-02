"""
Tree approximation module using stochastic approximation.

This module implements the stochastic approximation algorithm for generating
scenario trees that approximate a given stochastic process.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable
from .tree_structure import Tree, leaves, nodes, root, build_probabilities, height


def tree_approximation(
    tree: Tree,
    path: Callable[[], npt.NDArray[np.float64]],
    n_iterations: int,
    p: int = 2,
    r: int = 2
) -> Tree:
    """
    Approximate a stochastic process with a scenario tree using stochastic approximation.
    
    This function uses an iterative stochastic approximation procedure to generate
    a scenario tree that approximates a given stochastic process. At each iteration,
    a new sample path is generated and used to update the tree's node states and
    probabilities to better approximate the process.
    
    **Note**: This implementation now supports multi-dimensional trees (dimension > 1),
    enabling scenario trees for multiple correlated products/assets in a single structure.
    
    Parameters
    ----------
    tree : Tree
        Initial tree structure with specified branching pattern.
        The tree's states and probabilities will be updated in-place.
        Can have dimension > 1 for multi-product scenarios.
    path : callable
        Function that generates sample paths from the stochastic process.
        Must return array of shape (n_stages, dimension) where n_stages = height + 1
        and dimension matches tree.state.shape[1].
    n_iterations : int
        Number of iterations for the stochastic approximation procedure.
        More iterations generally lead to better approximation.
        Recommend 100k+ for multi-dimensional trees (dimension > 1).
    p : int, optional
        Choice of norm for distance calculation:
        - 0: max norm
        - 1: sum norm (Manhattan)
        - 2: Euclidean norm (default)
    r : int, optional
        Transportation distance parameter, default 2.
        Used in the multistage distance calculation.
        
    Returns
    -------
    Tree
        The input tree with updated states and probabilities that approximate
        the stochastic process. The tree's name is updated to include the
        approximation distance.
        
    Notes
    -----
    The algorithm uses:
    1. Critical probability handling to prevent branch loss
    2. Stochastic gradient descent with adaptive step size
    3. Multistage distance metric to measure approximation quality
    4. Proper multi-dimensional gradient computation (fixed in this version)
    
    Examples
    --------
    Single-dimensional tree:
    >>> from scentrees import Tree, tree_approximation, gaussian_path1d
    >>> tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
    >>> approximated = tree_approximation(tree, gaussian_path1d, 100000, p=2, r=2)
    >>> print(approximated.name)
    Tree 1x2x2x2 with d=... at 100000 iterations
    
    Multi-dimensional tree (e.g., 4 products):
    >>> tree_4d = Tree.from_branching([1, 3, 3, 3], dimension=4)
    >>> def multi_product_path():
    ...     # Return shape (4, 4) for 4 stages, 4 products
    ...     return np.random.randn(4, 4).cumsum(axis=0)
    >>> approximated_4d = tree_approximation(tree_4d, multi_product_path, 150000)
    >>> print(approximated_4d.state.shape)  # (40, 4) nodes
    
    References
    ----------
    Pflug, Georg Ch., and Alois Pichler. "Dynamic Generation of Scenario Trees."
    Computational Optimization and Applications 62.3 (2015): 641-668.
    """
    leaf_nodes, omegas, prob_leaf = leaves(tree)
    dm = tree.state.shape[1]  # dimension of states
    T = height(tree)  # tree height (number of stages - 1)
    n = len(leaf_nodes)  # number of leaves
    
    # FIX: Distance should be scalar per leaf, not vector (for multi-dimensional support)
    d = np.zeros((1, len(leaf_nodes)), dtype=np.float64)
    samplepath = np.zeros((T + 1, dm), dtype=np.float64)
    prob_leaf = np.zeros_like(prob_leaf)
    
    all_nodes = nodes(tree)
    path_to_leaves = [root(tree, i) for i in leaf_nodes]
    path_to_all_nodes = [root(tree, j) for j in all_nodes]
    
    for k in range(n_iterations):
        # Critical probability handling to prevent branch loss
        critical = max(0.0, 0.2 * np.sqrt(k) - 0.1 * n)
        tmp = np.where(prob_leaf <= critical)[0]
        
        # Generate new sample path
        samplepath = path()
        
        # Handle critical probabilities
        if len(tmp) > 0:
            prob_node = np.zeros(len(all_nodes), dtype=np.float64)
            prob_node[leaf_nodes] = prob_leaf
            
            # Accumulate probabilities up the tree
            for i in leaf_nodes:
                current = i
                while tree.parent[current] >= 0:
                    prob_node[tree.parent[current]] += prob_node[current]
                    current = tree.parent[current]
            
            # Reinitialize nodes with critical probabilities
            for tmp_i in tmp:
                rt = path_to_leaves[tmp_i]
                critical_nodes = np.where(prob_node[rt] <= critical)[0]
                tree.state[np.array(rt)[critical_nodes], :] = samplepath[critical_nodes, :]
        
        # Stochastic approximation step: find closest path in tree
        endleaf = 0  # Start from root (index 0)
        for t in range(T + 1):
            tmpleaves = tree.children[endleaf] if endleaf < len(tree.children) else []
            
            if len(tmpleaves) == 0:
                break
                
            disttemp = np.inf
            
            for i in tmpleaves:
                # Calculate distance from sample path to this tree path
                tree_path = tree.state[path_to_all_nodes[i], :]
                sample_segment = samplepath[:len(tree_path), :]
                
                if p == 0:  # Max norm
                    dist = np.max(np.abs(sample_segment - tree_path))
                elif p == 1:  # Manhattan norm
                    dist = np.sum(np.abs(sample_segment - tree_path))
                else:  # Euclidean or general p-norm
                    dist = np.linalg.norm(sample_segment - tree_path, ord=p)
                
                if dist < disttemp:
                    disttemp = dist
                    endleaf = i
        
        # Update probabilities
        istar = np.where(leaf_nodes == endleaf)[0]
        prob_leaf[istar] += 1.0
        
        # Update states using stochastic gradient
        tree_path_indices = path_to_leaves[endleaf - (leaf_nodes[0])]
        delta = tree.state[tree_path_indices, :] - samplepath[:len(tree_path_indices), :]
        
        # Update distance metric (FIX: compute total norm for multi-dimensional support)
        d[0, istar] += np.linalg.norm(delta, ord=p) ** r
        
        # Calculate gradient (FIX: handle multi-dimensional case)
        if dm == 1:
            # Original logic works fine for 1D
            norm_delta = np.linalg.norm(delta, ord=p, axis=0, keepdims=True)
            norm_delta = np.where(norm_delta > 0, norm_delta, 1.0)
            gradient = (r * norm_delta ** (r - p) *
                       np.abs(delta) ** (p - 1) * np.sign(delta))
        else:
            # For multi-dimensional, compute gradient element-wise
            total_norm = np.linalg.norm(delta, ord=p)
            if total_norm > 0:
                if p == 2:
                    # For Euclidean norm, gradient is simpler
                    gradient = r * (total_norm ** (r - 2)) * delta
                else:
                    # General case
                    element_contribution = np.abs(delta) ** (p - 1) * np.sign(delta)
                    norm_factor = total_norm ** (p - 1) if total_norm > 1e-10 else 1.0
                    gradient = r * (total_norm ** (r - p)) * element_contribution / norm_factor
            else:
                gradient = np.zeros_like(delta)
        
        # Adaptive step size
        ak = 1.0 / (30.0 + prob_leaf[istar])
        
        # Update tree states
        tree.state[tree_path_indices, :] -= gradient * ak
    
    # Calculate final probabilities and distance
    probabilities = prob_leaf / np.sum(prob_leaf)
    t_dist = (d @ probabilities.reshape(-1, 1) / n_iterations) ** (1.0 / r)
    
    # Update tree name with distance information
    tree.name = f"{tree.name} with d={t_dist.flatten()} at {n_iterations} iterations"
    
    # Build complete probability tree from leaf probabilities
    tree.probability = build_probabilities(tree, probabilities.reshape(-1, 1))
    
    return tree


__all__ = ['tree_approximation']