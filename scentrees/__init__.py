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
"""

__version__ = "0.1.0"
__author__ = "Python port from ScenTrees.jl"

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
]