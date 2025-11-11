"""
ScenTrees - Scenario trees and lattices for multistage stochastic optimization.

This is a Python port of the ScenTrees.jl Julia package.

The package provides:
- Generation of scenario trees and lattices using stochastic approximation
- Conditional density estimation for trajectory generation from data
- Visualization tools for trees and lattices
- Sample stochastic process generators

Main Components
---------------
Tree, Lattice : Data structures
    Core data structures for scenario trees and lattices
    
tree_approximation, lattice_approximation : Functions
    Main algorithms for approximating stochastic processes
    
gaussian_path1d, gaussian_path2d, running_maximum1d, running_maximum2d, path : Functions
    Sample stochastic process generators
    
kernel_scenarios : Function
    Conditional density estimation for generating trajectories from data
    
tree_plot, plot_lattice, bushiness_nesdistance : Functions
    Visualization utilities
    
set_seed : Function
    Set random seed for deterministic/reproducible results

Examples
--------
Generate and plot a scenario tree:

>>> from scentrees import Tree, tree_approximation, gaussian_path1d, tree_plot
>>> tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
>>> approximated = tree_approximation(tree, gaussian_path1d, 100000, p=2, r=2)
>>> tree_plot(approximated)

Generate a scenario lattice:

>>> from scentrees import lattice_approximation, running_maximum1d, plot_lattice
>>> lattice = lattice_approximation([1, 2, 3, 4], running_maximum1d, 100000, r=2, dimension=1)
>>> plot_lattice(lattice)

Generate trajectories from data:

>>> import numpy as np
>>> from scentrees import kernel_scenarios, tree_approximation, Tree
>>> data = np.array([gaussian_path1d()[:, 0] for _ in range(1000)])
>>> gen = kernel_scenarios(data, markovian=False)
>>> tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
>>> approximated = tree_approximation(tree, gen, 100000)

Reproducible results with seed:

>>> import scentrees
>>> scentrees.set_seed(42)  # Set seed for all RNG operations
>>> tree1 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
>>> path1 = scentrees.gaussian_path1d()
>>> # Reset seed to get identical results
>>> scentrees.set_seed(42)
>>> tree2 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
>>> path2 = scentrees.gaussian_path1d()
>>> # tree1 and tree2 will be identical
"""

__version__ = "0.1.0"
__author__ = "Python port from ScenTrees.jl"

import numpy as np

# Import all exports from submodules
from .tree_structure import (
    Tree,
    stage,
    height,
    leaves,
    nodes,
    root,
    part_tree,
    build_probabilities,
    tree_plot,
    plot_hd
)

from .tree_approximation import tree_approximation

from .lattice_approximation import (
    Lattice,
    lattice_approximation,
    plot_lattice
)

from .stochastic_paths import (
    gaussian_path1d,
    gaussian_path2d,
    running_maximum1d,
    running_maximum2d,
    path,
    set_random_seed
)

from .kernel_density import kernel_scenarios, create_multidim_generator

from .plotting_utils import bushiness_nesdistance


def set_seed(seed: int) -> None:
    """
    Set random seed for all RNG operations in the scentrees package.
    
    This function sets the random seed for all random number generators used
    throughout the package, ensuring deterministic and reproducible results
    across multiple runs. It affects:
    
    - Tree structure initialization (Tree.from_branching)
    - Stochastic path generators (gaussian_path1d, gaussian_path2d, etc.)
    - Kernel density estimation (kernel_scenarios, create_multidim_generator)
    - All approximation algorithms that use random sampling
    
    Parameters
    ----------
    seed : int
        Random seed value. Using the same seed will produce identical results
        across multiple runs of the same code.
        
    Examples
    --------
    Basic usage for reproducible tree generation:
    
    >>> import scentrees
    >>> import numpy as np
    >>>
    >>> # First run
    >>> scentrees.set_seed(42)
    >>> tree1 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
    >>> path1 = scentrees.gaussian_path1d()
    >>>
    >>> # Second run with same seed
    >>> scentrees.set_seed(42)
    >>> tree2 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
    >>> path2 = scentrees.gaussian_path1d()
    >>>
    >>> # Results are identical
    >>> assert np.allclose(tree1.state, tree2.state)
    >>> assert np.allclose(path1, path2)
    
    Reproducible tree approximation:
    
    >>> scentrees.set_seed(123)
    >>> tree = scentrees.Tree.from_branching([1, 3, 3, 3], dimension=1)
    >>> approximated1 = scentrees.tree_approximation(
    ...     tree, scentrees.gaussian_path1d, 10000
    ... )
    >>>
    >>> # Reset seed for identical approximation
    >>> scentrees.set_seed(123)
    >>> tree = scentrees.Tree.from_branching([1, 3, 3, 3], dimension=1)
    >>> approximated2 = scentrees.tree_approximation(
    ...     tree, scentrees.gaussian_path1d, 10000
    ... )
    >>>
    >>> assert np.allclose(approximated1.state, approximated2.state)
    
    Reproducible kernel density estimation:
    
    >>> import numpy as np
    >>> scentrees.set_seed(456)
    >>> # Generate sample data
    >>> data = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
    >>> gen1 = scentrees.kernel_scenarios(data, markovian=False)
    >>> trajectory1 = gen1()
    >>>
    >>> # Reset for identical trajectory
    >>> scentrees.set_seed(456)
    >>> data = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
    >>> gen2 = scentrees.kernel_scenarios(data, markovian=False)
    >>> trajectory2 = gen2()
    >>>
    >>> assert np.allclose(trajectory1, trajectory2)
    
    Notes
    -----
    - This function must be called before any operations that involve randomness
    - For reproducible results in parallel or multi-threaded environments,
      additional care may be needed
    - The default seed (if set_seed is never called) is 1012019
    
    See Also
    --------
    stochastic_paths.set_random_seed : Legacy function (deprecated, use set_seed instead)
    """
    # Import modules to access their _rng objects
    from . import stochastic_paths, kernel_density, tree_structure
    
    # Set random state for all modules
    stochastic_paths._rng = np.random.RandomState(seed)
    kernel_density._rng = np.random.RandomState(seed)
    tree_structure._rng = np.random.RandomState(seed)


# Define what gets exported with "from scentrees import *"
__all__ = [
    # Core data structures
    'Tree',
    'Lattice',
    
    # Main approximation functions
    'tree_approximation',
    'lattice_approximation',
    
    # Tree helper functions
    'stage',
    'height',
    'leaves',
    'nodes',
    'root',
    'part_tree',
    'build_probabilities',
    
    # Plotting functions
    'tree_plot',
    'plot_hd',
    'plot_lattice',
    'bushiness_nesdistance',
    
    # Stochastic path generators
    'gaussian_path1d',
    'gaussian_path2d',
    'running_maximum1d',
    'running_maximum2d',
    'path',
    'set_random_seed',
    
    # Kernel density estimation
    'kernel_scenarios',
    'create_multidim_generator',
    
    # Random seed management
    'set_seed',
]