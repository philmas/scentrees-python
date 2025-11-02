"""
Lattice approximation module for Markov processes.

This module implements scenario lattice generation for approximating
Markovian stochastic processes using stochastic approximation.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Callable, List, Union
from dataclasses import dataclass


@dataclass
class Lattice:
    """
    Scenario lattice data structure for Markov processes.
    
    A scenario lattice is a natural discretization of Markov processes where
    at each stage, nodes can transition to any node in the next stage (unlike
    trees where transitions are restricted to specific children).
    
    Attributes
    ----------
    name : str
        Name/description of the lattice
    state : list of ndarray
        List of state arrays, one per stage. Each array has shape
        (n_nodes_at_stage, 1, dimension)
    probability : list of ndarray
        List of probability arrays. probability[t] has shape
        (n_nodes_at_t-1, n_nodes_at_t, dimension) for t > 0
        
    Examples
    --------
    >>> from scentrees import lattice_approximation, gaussian_path1d
    >>> lattice = lattice_approximation([1, 2, 3, 4], gaussian_path1d, 100000)
    >>> len(lattice.state)
    4
    """
    name: str
    state: List[npt.NDArray[np.float64]]
    probability: List[npt.NDArray[np.float64]]


def lattice_approximation(
    branching: List[int],
    path: Callable[[], npt.NDArray[np.float64]],
    n_iterations: int,
    r: int = 2,
    dimension: int = 1
) -> Union[Lattice, List[Lattice]]:
    """
    Approximate a Markov process with a scenario lattice using stochastic approximation.
    
    Scenario lattices are particularly well-suited for approximating Markovian
    processes because they preserve the Markov property - at each stage, transitions
    can occur to any node in the next stage.
    
    Parameters
    ----------
    branching : list of int
        Branching structure defining nodes at each stage.
        Example: [1, 2, 3, 4] creates a lattice with 1, 2, 3, 4 nodes at stages 0-3
    path : callable
        Function generating sample paths from the stochastic process.
        Must return array of shape (len(branching), dimension)
    n_iterations : int
        Number of iterations for stochastic approximation.
        More iterations generally lead to better approximation.
    r : int, optional
        Transportation distance parameter, default 2
    dimension : int, optional
        Dimension of the state space, default 1.
        If dimension > 1, returns a list of lattices (one per dimension)
        
    Returns
    -------
    Lattice or list of Lattice
        If dimension=1, returns a single Lattice.
        If dimension>1, returns a list of Lattice objects, one per dimension.
        
    Notes
    -----
    The algorithm:
    1. Initializes lattice with random states
    2. For each iteration:
       - Generates a new sample path
       - Finds closest lattice entry at each stage
       - Updates states using stochastic gradient
       - Handles nodes with low visitation probability
    3. Normalizes probabilities and calculates distance metric
    
    Examples
    --------
    >>> from scentrees import lattice_approximation, gaussian_path1d
    >>> lattice = lattice_approximation([1, 2, 3, 4], gaussian_path1d, 100000, r=2, dimension=1)
    >>> print(lattice.name)
    Approximated Lattice with [1, 2, 3, 4] branching structure...
    
    >>> # For 2D processes
    >>> from scentrees import gaussian_path2d
    >>> lattices = lattice_approximation([1, 2, 3, 4], gaussian_path2d, 100000, dimension=2)
    >>> len(lattices)
    2
    
    References
    ----------
    Pflug, Georg Ch., and Alois Pichler. "Dynamic Generation of Scenario Trees."
    Computational Optimization and Applications 62.3 (2015): 641-668.
    """
    tdist = np.zeros(dimension, dtype=np.float64)
    T = len(branching)
    
    # Initialize states and probabilities
    states = [np.zeros((branching[j], 1, dimension)) for j in range(T)]
    probabilities = [np.zeros((branching[0], 1, dimension))] + \
                   [np.zeros((branching[j-1], branching[j], dimension)) for j in range(1, T)]
    
    # Get initial sample path
    init_path = path()
    
    # Validate dimension
    if dimension != init_path.shape[1]:
        raise ValueError(
            f"Dimension of lattice ({dimension}) does not match "
            f"dimension of input array ({init_path.shape[1]})"
        )
    
    # Initialize states from initial path
    for t in range(T):
        states[t][:, :, :] = init_path[t, :]
    
    Z = np.zeros((T, dimension), dtype=np.float64)
    
    # Stochastic approximation loop
    for n in range(n_iterations):
        Z = path()  # Generate new sample path
        last_index = np.ones(dimension, dtype=np.int64)
        dist = np.zeros(dimension, dtype=np.float64)
        
        for t in range(T):
            for i in range(dimension):
                # Corrective action for lost nodes (nodes with very low probability)
                prob_sum = np.sum(probabilities[t][:, :, i], axis=1) if t > 0 else probabilities[t][:, 0, i]
                threshold = 1.3 * np.sqrt(n) / branching[t]
                low_prob_nodes = np.where(prob_sum < threshold)[0] if t > 0 else []
                
                if len(low_prob_nodes) > 0:
                    states[t][low_prob_nodes, :, i] = Z[t, i]
                
                # Find closest lattice entry
                distances = np.abs(states[t][:, 0, i] - Z[t, i])
                new_index = np.argmin(distances)
                min_dist = distances[new_index]
                
                dist[i] += min_dist ** 2
                
                # Update probability counter
                if t == 0:
                    probabilities[t][new_index, 0, i] += 1.0
                else:
                    probabilities[t][last_index[i], new_index, i] += 1.0
                
                # Stochastic approximation update for states
                step_size = 1.0 / (Z[0, 0] + 30 + n)
                gradient = r * (min_dist ** (r - 1)) * np.sign(states[t][new_index, 0, i] - Z[t, i])
                states[t][new_index, 0, i] -= gradient * step_size
                
                last_index[i] = new_index
        
        # Update multistage distance
        dist = dist ** 0.5
        tdist = (tdist * (n - 1) + dist ** r) / n
    
    # Normalize probabilities
    probabilities = [p / n_iterations for p in probabilities]
    
    # Round results for cleaner output
    states = [np.round(s, decimals=6) for s in states]
    probabilities = [np.round(p, decimals=6) for p in probabilities]
    
    # Return appropriate result based on dimension
    if dimension == 1:
        return Lattice(
            name=f"Approximated Lattice with {branching} branching structure and "
                 f"distance={tdist ** (1/r)} at {n_iterations} iterations",
            state=states,
            probability=probabilities
        )
    else:
        # For multi-dimensional, split into separate lattices per dimension
        lattices = []
        for i in range(dimension):
            st = [np.zeros((branching[j], 1, 1)) for j in range(T)]
            pp = [np.zeros((branching[0], 1, 1))] + \
                 [np.zeros((branching[j-1], branching[j], 1)) for j in range(1, T)]
            
            for j in range(T):
                st[j][:, :, 0] = states[j][:, :, i]
                pp[j][:, :, 0] = probabilities[j][:, :, i]
            
            sublattice = Lattice(
                name=f"Lattice {i} with distance={tdist[i] ** (1/r)}",
                state=st,
                probability=pp
            )
            lattices.append(sublattice)
        
        return lattices


def plot_lattice(lattice: Lattice, fig: Union[int, tuple, None] = None) -> None:
    """
    Plot a scenario lattice with probability density.
    
    Parameters
    ----------
    lattice : Lattice
        The scenario lattice to plot
    fig : int, tuple, or None, optional
        Figure specification. Can be:
        - int: figure number
        - tuple: figure size (width, height)
        - None: use default
        
    Examples
    --------
    >>> from scentrees import lattice_approximation, gaussian_path1d, plot_lattice
    >>> lattice = lattice_approximation([1, 2, 3, 4], gaussian_path1d, 100000)
    >>> plot_lattice(lattice)
    """
    if fig is not None:
        if isinstance(fig, tuple):
            plt.figure(figsize=fig)
        else:
            plt.figure(figsize=(6, 4))
    
    # Main lattice plot
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax1.set_title("states")
    ax1.set_xlabel("stage, time", fontsize=11)
    ax1.set_xticks(range(1, len(lattice.state) + 1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot lattice transitions
    for t in range(1, len(lattice.state)):
        for i in range(len(lattice.state[t-1])):
            for j in range(len(lattice.state[t])):
                ax1.plot(
                    [t, t+1],
                    [lattice.state[t-1][i, 0, 0], lattice.state[t][j, 0, 0]],
                    'b-', alpha=0.6
                )
    
    # Probability density plot
    ax2 = plt.subplot2grid((1, 4), (0, 3))
    ax2.set_title("probabilities")
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([])
    
    # Use terminal stage for density estimation
    stts = lattice.state[-1][:, 0, 0]
    n = len(stts)
    h = 1.05 * np.std(stts) / (n ** 0.2) + np.finfo(float).eps  # Silverman's rule
    
    # Calculate marginal probabilities at terminal stage
    proba = np.sum(lattice.probability[-1], axis=0).flatten()
    
    t = np.linspace(stts.min() - h, stts.max() + h, 100)
    density = np.zeros_like(t)
    
    # Triweight kernel density estimation
    for i, ti in enumerate(t):
        for j, xj in enumerate(stts):
            tmp = (xj - ti) / h
            density[i] += proba[j] * (35/32) * max(1.0 - tmp**2, 0.0)**3 / h
    
    ax2.plot(density, t)
    ax2.fill_betweenx(t, 0, density, alpha=0.3)
    
    plt.tight_layout()


__all__ = ['Lattice', 'lattice_approximation', 'plot_lattice']