"""
Tree structure module for scenario trees.

This module defines the Tree class and associated helper functions for
working with scenario trees in multistage stochastic optimization.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass, field
from scipy import stats

# Global random state for reproducibility
_rng = np.random.RandomState(1012019)


@dataclass
class Tree:
    """
    Scenario tree data structure.
    
    A scenario tree represents discrete approximations of stochastic processes
    over multiple stages. Each node has a state value and a probability of
    transitioning to its child nodes.
    
    Attributes
    ----------
    name : str
        Name/description of the tree
    parent : ndarray
        Array where parent[i] gives the parent node index of node i.
        Root node has parent -1 (Julia uses 0, we use -1 for "no parent")
    children : list of lists
        children[i] contains list of child node indices for node i
    state : ndarray
        State values at each node, shape (n_nodes, dimension)
    probability : ndarray
        Transition probabilities, shape (n_nodes, 1)
    
    Examples
    --------
    >>> tree = Tree([1, 2, 2, 2], dimension=1)
    >>> tree.name
    'Tree 1x2x2x2'
    >>> height(tree)
    3
    """
    name: str
    parent: npt.NDArray[np.int64]
    children: List[List[int]] = field(default_factory=list)
    state: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    probability: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Compute children after initialization."""
        if len(self.children) == 0:
            self.children = _compute_children(self.parent)
    
    @classmethod
    def from_branching(cls, branching: List[int], dimension: int = 1) -> 'Tree':
        """
        Create a tree from branching structure.
        
        Parameters
        ----------
        branching : list of int
            Branching structure, e.g., [1, 2, 2, 2] for binary tree
            Must start with 1 for root node
        dimension : int, optional
            Dimension of state space, default 1
            
        Returns
        -------
        Tree
            Initialized tree with random states and probabilities
            
        Examples
        --------
        >>> tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
        >>> len(tree.parent)
        15
        """
        if branching[0] != 1:
            raise ValueError("Branching structure must start with 1 (root node)")
        
        name = f"Tree {branching[0]}"
        parent = np.array([-1], dtype=np.int64)  # Root has no parent
        state = _rng.randn(1, dimension)
        probability = np.ones((1, 1), dtype=np.float64)
        
        leaves = 1
        for stage_idx in range(1, len(branching)):
            b = branching[stage_idx]
            leaves = leaves * branching[stage_idx - 1]
            
            # Create new nodes for this stage
            # Each of the previous 'leaves' nodes gets 'b' children
            new_parent_indices = np.repeat(
                np.arange(len(parent) - leaves, len(parent)),
                b
            )
            parent = np.concatenate([parent, new_parent_indices])
            
            # Random states for new nodes
            new_states = _rng.randn(len(new_parent_indices), dimension)
            state = np.vstack([state, new_states])
            
            # Random probabilities (will be normalized per parent)
            tmp = _rng.uniform(0.3, 1.0, size=(b, leaves))
            tmp = tmp / tmp.sum(axis=0, keepdims=True)
            new_probs = tmp.T.flatten().reshape(-1, 1)
            probability = np.vstack([probability, new_probs])
            
            name = f"{name}x{b}"
        
        tree = cls(
            name=name,
            parent=parent,
            children=[],
            state=state,
            probability=probability
        )
        tree.children = _compute_children(parent)
        return tree
    
    @classmethod
    def from_identifier(cls, identifier: int) -> 'Tree':
        """
        Create predefined example trees.
        
        Parameters
        ----------
        identifier : int
            Tree identifier (0, 302, 303, 304, 305, 306, 307, 401, 402, 404, 405)
            
        Returns
        -------
        Tree
            Predefined tree
            
        Examples
        --------
        >>> tree = Tree.from_identifier(402)
        >>> tree.name
        'Tree 1x2x2x2'
        """
        if identifier == 0:
            return cls(
                name="Empty Tree",
                parent=np.array([], dtype=np.int64),
                state=np.array([]).reshape(0, 1),
                probability=np.array([]).reshape(0, 1)
            )
        elif identifier == 302:
            return cls(
                name="Tree 1x2x2",
                parent=np.array([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64),
                state=np.array([2.0, 2.1, 1.9, 4.0, 1.0, 3.0, 0.0]).reshape(-1, 1),
                probability=np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
            )
        elif identifier == 303:
            return cls(
                name="Tree 1x1x4",
                parent=np.array([-1, 0, 1, 1, 1, 1], dtype=np.int64),
                state=np.array([3.0, 3.0, 6.0, 4.0, 2.0, 0.0]).reshape(-1, 1),
                probability=np.array([1.0, 1.0, 0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)
            )
        elif identifier == 304:
            return cls(
                name="Tree 1x4x1x1",
                parent=np.array([-1, 0, 1, -1, 3, 4, -1, 6, 7, -1, 9, 10], dtype=np.int64),
                state=np.array([0.1, 2.1, 3.0, 0.1, 1.9, 1.0, 0.0, -2.9, -1.0, -0.1, -3.1, -4.0]).reshape(-1, 1),
                probability=np.array([0.14, 1.0, 1.0, 0.06, 1.0, 1.0, 0.48, 1.0, 1.0, 0.32, 1.0, 1.0]).reshape(-1, 1)
            )
        elif identifier == 305:
            return cls(
                name="Tree 1x1x4",
                parent=np.array([-1, 0, 1, 1, 1, 1], dtype=np.int64),
                state=np.array([0.0, 10.0, 28.0, 22.0, 21.0, 20.0]).reshape(-1, 1),
                probability=np.array([1.0, 1.0, 0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)
            )
        elif identifier == 306:
            return cls(
                name="Tree 1x2x2",
                parent=np.array([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64),
                state=np.array([0.0, 10.0, 10.0, 28.0, 22.0, 21.0, 20.0]).reshape(-1, 1),
                probability=np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
            )
        elif identifier == 307:
            return cls(
                name="Tree 1x4x1",
                parent=np.array([-1, 0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int64),
                state=np.array([0.0, 10.0, 10.0, 10.0, 10.0, 28.0, 22.0, 21.0, 20.0]).reshape(-1, 1),
                probability=np.array([1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
            )
        elif identifier == 401:
            return cls(
                name="Tree 1x1x2x2",
                parent=np.array([-1, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
                state=np.array([10.0, 10.0, 8.0, 12.0, 9.0, 6.0, 10.0, 13.0]).reshape(-1, 1),
                probability=np.array([1.0, 1.0, 0.66, 0.34, 0.24, 0.76, 0.46, 0.54]).reshape(-1, 1)
            )
        elif identifier == 402:
            return cls(
                name="Tree 1x2x2x2",
                parent=np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64),
                state=np.array([10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 5.0, 18.0, 16.0, 13.0, 11.0, 10.0, 7.0, 6.0, 3.0]).reshape(-1, 1),
                probability=np.array([1.0, 0.8, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.2, 0.5, 0.5, 0.4, 0.6, 0.7, 0.3]).reshape(-1, 1)
            )
        elif identifier == 404:
            return cls(
                name="Tree 1x2x2x2",
                parent=np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64),
                state=(0.2 + 0.6744) * np.array([0.0, 1.0, -1.0, 2.0, 0.1, 0.0, -2.0, 3.0, 1.1, 0.9, -1.1, 1.2, -1.2, -0.8, -3.2]).reshape(-1, 1),
                probability=np.array([1.0, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.5, 0.5, 0.6, 0.4, 0.4, 0.6, 0.3, 0.7]).reshape(-1, 1)
            )
        elif identifier == 405:
            return cls(
                name="Tree 5",
                parent=np.array([-1, 0, 1, 2, 2, 1, 5, 5, 5, 0, 9, 9, 10, 11, 11, 11], dtype=np.int64),
                state=np.array([7.0, 12.0, 14.0, 15.0, 13.0, 9.0, 11.0, 10.0, 8.0, 4.0, 6.0, 2.0, 5.0, 0.0, 1.0, 3.0]).reshape(-1, 1),
                probability=np.array([1.0, 0.7, 0.4, 0.7, 0.3, 0.6, 0.2, 0.3, 0.5, 0.3, 0.2, 0.8, 1.0, 0.3, 0.5, 0.2]).reshape(-1, 1)
            )
        else:
            raise ValueError(f"Unknown tree identifier: {identifier}")


def _compute_children(parent: npt.NDArray[np.int64]) -> List[List[int]]:
    """
    Compute children list from parent array.
    
    Parameters
    ----------
    parent : ndarray
        Parent array where parent[i] is the parent of node i
        
    Returns
    -------
    list of lists
        children[i] contains indices of children of node i
    """
    unique_parents = np.unique(parent[parent >= 0])
    children = []
    for node in unique_parents:
        children.append(np.where(parent == node)[0].tolist())
    return children


def stage(tree: Tree, node: Optional[Union[int, List[int], npt.NDArray[np.int64]]] = None) -> Union[int, npt.NDArray[np.int64]]:
    """
    Return the stage of each node in the tree.
    
    The stage is the distance from the root (root is at stage 0).
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
    node : int, list, or ndarray, optional
        Node index or indices. If None, returns stages for all nodes
        
    Returns
    -------
    int or ndarray
        Stage number(s)
        
    Examples
    --------
    >>> tree = Tree.from_identifier(402)
    >>> stage(tree, 0)  # Root node
    0
    >>> stage(tree, 7)  # A leaf node
    3
    """
    if node is None:
        node = np.arange(len(tree.parent))
    elif isinstance(node, int):
        node = np.array([node])
    elif isinstance(node, list):
        node = np.array(node)
    
    stages = np.zeros(len(node), dtype=np.int64)
    for i, n in enumerate(node):
        pred = n
        while pred >= 0 and tree.parent[pred] >= 0:
            pred = tree.parent[pred]
            stages[i] += 1
    
    return stages[0] if len(stages) == 1 else stages


def height(tree: Tree) -> int:
    """
    Return the height of the tree (maximum stage number).
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
        
    Returns
    -------
    int
        Tree height
        
    Examples
    --------
    >>> tree = Tree.from_branching([1, 2, 2, 2])
    >>> height(tree)
    3
    """
    return int(np.max(stage(tree)))


def leaves(tree: Tree, node: Optional[Union[int, List[int]]] = None) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Return leaf nodes, their indices, and conditional probabilities.
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
    node : int or list, optional
        If provided, returns only leaves descending from these nodes
        
    Returns
    -------
    leaves : ndarray
        Indices of leaf nodes
    omegas : ndarray
        Indices in the original leaf array
    prob : ndarray
        Probabilities of reaching each leaf
        
    Examples
    --------
    >>> tree = Tree.from_identifier(402)
    >>> leaf_nodes, omegas, probs = leaves(tree)
    >>> len(leaf_nodes)
    8
    >>> np.isclose(probs.sum(), 1.0)
    True
    """
    nodes_array = np.arange(len(tree.parent))
    # Leaves are nodes that are not parents of any other node
    leaf_nodes = np.setdiff1d(nodes_array, tree.parent[tree.parent >= 0])
    omegas = np.arange(len(leaf_nodes))
    
    if node is not None:
        if isinstance(node, int):
            node = [node]
        if -1 not in node and len(node) > 0:
            # Filter leaves to only those descending from specified nodes
            omegas_set = set()
            current_nodes = leaf_nodes.copy()
            while np.any(current_nodes >= 0):
                for idx, n in enumerate(current_nodes):
                    if n in node:
                        omegas_set.add(idx)
                current_nodes = np.array([tree.parent[max(0, n)] if n >= 0 else -1 
                                         for n in current_nodes])
            omegas = np.array(sorted(omegas_set))
    
    leaf_nodes = leaf_nodes[omegas]
    prob = np.ones(len(leaf_nodes), dtype=np.float64)
    
    # Calculate probability of reaching each leaf
    current_nodes = leaf_nodes.copy()
    while np.any(current_nodes >= 0):
        prob = prob * np.array([tree.probability[n, 0] if n >= 0 else 1.0 
                                for n in current_nodes])
        current_nodes = np.array([tree.parent[n] if n >= 0 else -1 
                                  for n in current_nodes])
    
    return leaf_nodes, omegas, prob


def nodes(tree: Tree, t: Optional[int] = None) -> npt.NDArray[np.int64]:
    """
    Return nodes in the tree, optionally filtered by stage.
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
    t : int, optional
        Stage number. If None, returns all nodes
        
    Returns
    -------
    ndarray
        Node indices
        
    Examples
    --------
    >>> tree = Tree.from_identifier(402)
    >>> len(nodes(tree))
    15
    >>> len(nodes(tree, t=0))  # Root only
    1
    """
    all_nodes = np.arange(len(tree.parent))
    if t is None:
        return all_nodes
    else:
        stg = stage(tree)
        return all_nodes[stg == t]


def root(tree: Tree, node: Optional[Union[int, List[int]]] = None) -> npt.NDArray[np.int64]:
    """
    Return the root or path from root to specified node(s).
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
    node : int or list, optional
        If None, returns root node. Otherwise returns path from root to node(s)
        
    Returns
    -------
    ndarray
        Root node index or path indices
        
    Examples
    --------
    >>> tree = Tree.from_identifier(402)
    >>> root(tree)
    array([0])
    >>> root(tree, 7)  # Path from root to node 7
    array([0, 1, 3, 7])
    """
    if node is None:
        return tree.children[0] if len(tree.children) > 0 else np.array([0])
    
    # Handle both Python int and numpy integer types
    if isinstance(node, (int, np.integer)):
        node = [int(node)]
    
    root_path = np.arange(len(tree.parent))
    for n in node:
        node_path = []
        tmp = n
        while tmp >= 0:
            node_path.append(tmp)
            tmp = tree.parent[tmp]
        node_path = node_path[::-1]  # Reverse to get root-to-node order
        root_path = np.array([i for i in node_path if i in root_path])
    
    return root_path


def part_tree(tree: Tree) -> List[Tree]:
    """
    Split multi-dimensional tree into separate trees for each dimension.
    
    Parameters
    ----------
    tree : Tree
        Multi-dimensional tree
        
    Returns
    -------
    list of Tree
        One tree for each dimension
        
    Examples
    --------
    >>> tree = Tree.from_branching([1, 2, 2], dimension=2)
    >>> trees = part_tree(tree)
    >>> len(trees)
    2
    """
    trees = []
    for col in range(tree.state.shape[1]):
        subtree = Tree(
            name=f"Tree of state {col}",
            parent=tree.parent.copy(),
            children=tree.children.copy(),
            state=tree.state[:, col:col+1].copy(),
            probability=tree.probability.copy()
        )
        trees.append(subtree)
    return trees


def build_probabilities(tree: Tree, probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Build probability tree from leaf probabilities.
    
    Parameters
    ----------
    tree : Tree
        The scenario tree
    probabilities : ndarray
        Leaf probabilities or all node probabilities
        
    Returns
    -------
    ndarray
        Complete probability array
        
    Examples
    --------
    >>> tree = Tree.from_identifier(402)
    >>> leaf_nodes, _, probs = leaves(tree)
    >>> build_probabilities(tree, probs.reshape(-1, 1))
    """
    leaf_nodes, omegas, prob_leaf = leaves(tree)
    
    if len(probabilities) == len(tree.parent):
        tree.probability = probabilities.copy()
    else:
        probabilities = np.maximum(0.0, probabilities)
        j = leaf_nodes.copy()
        i = tree.parent[j]
        mask = i >= 0
        j = j[mask]
        i = i[mask]
        
        tree.probability = np.zeros((len(tree.state), 1), dtype=np.float64)
        tree.probability[j] = probabilities.copy()
        
        while len(i) > 0:
            for k in range(len(i)):
                tree.probability[i[k]] += tree.probability[j[k]]
            
            # Normalize (suppress expected divide-by-zero warnings)
            with np.errstate(divide='ignore', invalid='ignore'):
                tree.probability[j] = np.where(
                    tree.probability[i] > 0,
                    tree.probability[j] / tree.probability[i],
                    0.0
                )
            tree.probability[np.isnan(tree.probability)] = 0.0
            
            j = np.unique(i)
            i = tree.parent[j]
            mask = i >= 0
            j = j[mask]
            i = i[mask]
    
    return tree.probability


def tree_plot(tree: Tree, fig: Optional[Union[int, Tuple[int, int]]] = None, dim_index: int = 0) -> None:
    """
    Plot the scenario tree with probability density.
    
    Parameters
    ----------
    tree : Tree
        The scenario tree to plot
    fig : int or tuple, optional
        Figure size specification
    dim_index : int, optional
        Which dimension to plot for multi-dimensional trees (default: 0).
        For a tree with dimension=4, use dim_index=0,1,2,3 to plot each product.
        
    Examples
    --------
    Single-dimensional tree:
    >>> tree = Tree.from_identifier(402)
    >>> tree_plot(tree)
    
    Multi-dimensional tree (e.g., 4 products):
    >>> tree_4d = Tree.from_branching([1, 3, 3], dimension=4)
    >>> tree_plot(tree_4d, dim_index=0)  # Plot first product (e.g., Chairs)
    >>> tree_plot(tree_4d, dim_index=1)  # Plot second product (e.g., Tables)
    >>> tree_plot(tree_4d, dim_index=2)  # Plot third product (e.g., Desks)
    >>> tree_plot(tree_4d, dim_index=3)  # Plot fourth product (e.g., Office Chairs)
    """
    # Validate dim_index
    n_dims = tree.state.shape[1]
    if dim_index < 0 or dim_index >= n_dims:
        raise ValueError(f"dim_index must be between 0 and {n_dims-1}, got {dim_index}")
    if fig is not None:
        if isinstance(fig, tuple):
            plt.figure(figsize=fig)
        else:
            plt.figure(fig)
    
    # Main tree plot
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax1.set_title("states")
    ax1.set_xlabel("stage, time", fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    stg = stage(tree)
    max_stage = height(tree)
    ax1.set_xticks(range(max_stage + 2))
    
    # Plot tree branches for specified dimension
    for i in range(len(tree.parent)):
        if stg[i] > 0:
            parent_idx = tree.parent[i]
            ax1.plot(
                [stg[i], stg[i] + 1],
                [tree.state[parent_idx, dim_index], tree.state[i, dim_index]],
                linewidth=1.5
            )
    
    # Probability density plot for specified dimension
    ax2 = plt.subplot2grid((1, 4), (0, 3))
    if n_dims > 1:
        ax2.set_title(f"probabilities (dim {dim_index})")
    else:
        ax2.set_title("probabilities")
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([])
    
    leaf_nodes, _, prob_leaf = leaves(tree)
    Yi = tree.state[leaf_nodes, dim_index]
    nY = len(Yi)
    h = 1.05 * np.std(Yi) / (nY ** 0.2) + 1e-3  # Silverman's rule of thumb
    
    t = np.linspace(Yi.min() - h, Yi.max() + h, 100)
    density = np.zeros_like(t)
    
    # Triweight kernel density estimation
    for i, ti in enumerate(t):
        for j, xj in enumerate(Yi):
            tmp = (xj - ti) / h
            density[i] += prob_leaf[j] * (35/32) * max(1.0 - tmp**2, 0.0)**3 / h
    
    ax2.plot(density, t)
    ax2.fill_betweenx(t, 0, density, alpha=0.3)
    
    plt.tight_layout()


def plot_hd(tree: Tree) -> None:
    """
    Plot multi-dimensional tree (one subplot per dimension).
    
    Parameters
    ----------
    tree : Tree
        Multi-dimensional scenario tree
        
    Examples
    --------
    >>> tree = Tree.from_branching([1, 2, 2], dimension=2)
    >>> plot_hd(tree)
    """
    n_dims = tree.state.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(10, 6))
    
    if n_dims == 1:
        axes = [axes]
    
    stg = stage(tree)
    
    for dim in range(n_dims):
        ax = axes[dim]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("stage, time")
        ax.set_ylabel("states")
        
        # Plot tree branches
        for i in range(len(tree.parent)):
            if stg[i] > 0:
                parent_idx = tree.parent[i]
                ax.plot(
                    [stg[i], stg[i] + 1],
                    [tree.state[parent_idx, dim], tree.state[i, dim]]
                )
        
        ax.set_xticks(np.unique(stg))
    
    plt.tight_layout()


__all__ = [
    'Tree',
    'stage',
    'height',
    'leaves',
    'nodes',
    'root',
    'part_tree',
    'build_probabilities',
    'tree_plot',
    'plot_hd',
]