"""
Plotting utilities for scenario trees and lattices.

This module provides additional plotting functions for visualizing
relationships between tree properties and approximation quality.
"""

import numpy as np
import matplotlib.pyplot as plt


def bushiness_nesdistance() -> None:
    """
    Plot the relationship between tree bushiness and nested distance.
    
    This function displays how the multistage distance varies with the
    bushiness (branching factor) of trees at different heights. The plot
    shows that as bushiness increases, the approximation quality improves
    (distance decreases), and this effect is more pronounced for taller trees.
    
    The data shown is from example calculations demonstrating the trade-off
    between tree complexity and approximation accuracy.
    
    Examples
    --------
    >>> from scentrees import bushiness_nesdistance
    >>> bushiness_nesdistance()
    >>> plt.show()
    
    Notes
    -----
    The plot shows:
    - X-axis: Branches at each node (bushiness)
    - Y-axis: Multistage distance (lower is better)
    - Multiple lines for different tree heights
    
    This helps in choosing appropriate branching structures for scenario trees
    based on desired approximation quality and computational constraints.
    """
    bushns = np.array([3, 4, 5, 6, 7, 8])
    nstdistance = np.array([
        [0.24868, 0.16772, 0.12339, 0.09586, 0.07725, 0.06408],
        [0.21781, 0.12874, 0.08541, 0.0611, 0.04575, 0.03653],
        [0.16249, 0.08449, 0.05089, 0.03337, 0.02382, 0.01724],
        [0.11346, 0.05236, 0.0289, 0.01813, 0.01188, 0.00855]
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i in range(nstdistance.shape[0]):
        ax.plot(
            bushns,
            nstdistance[i, :],
            linewidth=2,
            marker='.',
            markersize=10,
            label=f'Height = {i + 1}'
        )
    
    ax.set_xlabel('Branches at each node', fontsize=12)
    ax.set_ylabel('Multistage distance', fontsize=12)
    ax.set_title('Tree Bushiness vs Nested Distance', fontsize=14)
    ax.legend(loc='upper right', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


__all__ = ['bushiness_nesdistance']