"""
Stochastic path generators for scenario tree and lattice approximation.

This module provides sample path generators for various stochastic processes
that can be used with the tree and lattice approximation algorithms.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable

# Global random state for reproducibility (matching Julia's seed)
_rng = np.random.RandomState(1012019)


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducible results.
    
    Parameters
    ----------
    seed : int
        Random seed value
    
    Examples
    --------
    >>> set_random_seed(42)
    >>> path = gaussian_path1d()
    """
    global _rng
    _rng = np.random.RandomState(seed)


def gaussian_path1d() -> npt.NDArray[np.float64]:
    """
    Generate a 1-dimensional Gaussian random walk path.
    
    Returns a 4-stage path (4x1 array) starting from 0.
    
    Returns
    -------
    ndarray
        Array of shape (4, 1) representing a Gaussian random walk
    
    Examples
    --------
    >>> path = gaussian_path1d()
    >>> path.shape
    (4, 1)
    >>> path[0, 0]  # First value is always 0
    0.0
    """
    # Start at 0, then cumulative sum of 3 random normal values
    increments = _rng.randn(3, 1)
    path = np.vstack([np.array([[0.0]]), np.cumsum(increments, axis=0)])
    return path


def gaussian_path2d() -> npt.NDArray[np.float64]:
    """
    Generate a 2-dimensional Gaussian random walk path.
    
    Returns a 4-stage path (4x2 array) with correlated dimensions.
    
    Returns
    -------
    ndarray
        Array of shape (4, 2) representing a 2D Gaussian random walk
    
    Examples
    --------
    >>> path = gaussian_path2d()
    >>> path.shape
    (4, 2)
    >>> np.allclose(path[0, :], [0.0, 0.0])  # First values are 0
    True
    """
    # Create correlated random walk
    # Correlation matrix: [[1.0, 0.0], [0.9, 0.3]]
    gsmatrix = _rng.randn(4, 2) @ np.array([[1.0, 0.0], [0.9, 0.3]])
    gsmatrix[0, :] = 0.0
    # Add drift and cumsum
    path = np.cumsum(gsmatrix + np.array([1.0, 0.0]), axis=0)
    return path


def running_maximum1d() -> npt.NDArray[np.float64]:
    """
    Generate a 1-dimensional running maximum process.
    
    Returns a 4-stage path (4x1 array) where each value is the maximum
    of the current and all previous values of a Gaussian random walk.
    
    Returns
    -------
    ndarray
        Array of shape (4, 1) representing a running maximum process
    
    Examples
    --------
    >>> path = running_maximum1d()
    >>> path.shape
    (4, 1)
    >>> # Each value should be >= previous value
    >>> all(path[i] >= path[i-1] for i in range(1, 4))
    True
    """
    # Start with Gaussian random walk
    increments = _rng.randn(3, 1)
    rmatrix = np.vstack([np.array([[0.0]]), np.cumsum(increments, axis=0)])
    
    # Compute running maximum
    for i in range(1, 4):
        rmatrix[i] = np.maximum(rmatrix[i-1], rmatrix[i])
    
    return rmatrix


def running_maximum2d() -> npt.NDArray[np.float64]:
    """
    Generate a 2-dimensional running maximum process.
    
    Returns a 4-stage path (4x2 array) with correlated dimensions,
    where each dimension follows a running maximum process.
    
    Returns
    -------
    ndarray
        Array of shape (4, 2) representing a 2D running maximum process
    
    Examples
    --------
    >>> path = running_maximum2d()
    >>> path.shape
    (4, 2)
    """
    # Start with Gaussian random walk for first dimension
    increments = _rng.randn(3, 1)
    rmatrix = np.vstack([np.array([[0.0]]), np.cumsum(increments, axis=0)])
    
    # Create 2D matrix
    rmatrix2d = np.zeros((4, 2))
    rmatrix2d[:, 0] = rmatrix[:, 0]
    
    # Apply running maximum to second dimension
    for j in range(1, 2):  # Only second column (j=1)
        for i in range(1, 4):
            rmatrix2d[i, j] = np.maximum(rmatrix[i-1], rmatrix[i])
    
    # Apply correlation matrix
    correlation_matrix = np.array([[1.0, 0.0], [0.9, 0.3]])
    return rmatrix2d @ correlation_matrix


def path() -> npt.NDArray[np.float64]:
    """
    Generate a simple stock price path.
    
    Returns a 4-stage path (4x1 array) representing stock prices
    following a simple random walk starting at 100.
    
    Returns
    -------
    ndarray
        Array of shape (4, 1) representing stock prices
    
    Examples
    --------
    >>> prices = path()
    >>> prices.shape
    (4, 1)
    >>> prices[0, 0]  # First price
    100.0
    """
    increments = _rng.randn(3, 1)
    stock_path = 100.0 + 50.0 * np.vstack([np.array([[0.0]]), np.cumsum(increments, axis=0)])
    return stock_path


# Type alias for path generator functions
PathGenerator = Callable[[], npt.NDArray[np.float64]]


__all__ = [
    'set_random_seed',
    'gaussian_path1d',
    'gaussian_path2d', 
    'running_maximum1d',
    'running_maximum2d',
    'path',
    'PathGenerator',
]