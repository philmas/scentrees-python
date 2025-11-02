# ScenTrees (Python)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python port of the [ScenTrees.jl](https://github.com/kirui93/ScenTrees.jl) Julia package for generating and improving scenario trees and scenario lattices for multistage stochastic optimization problems using _stochastic approximation_.

**Note:** See also the active Julia fork at https://github.com/aloispichler/ScenTrees.jl.

## Overview

`scentrees` is a Python package that provides functions for generating scenario trees and scenario lattices from stochastic processes and stochastic data. These structures are essential for solving multistage stochastic optimization problems.

### Key Features

- **Scenario Tree Generation**: Approximate discrete-time stochastic processes with scenario trees using stochastic approximation
- **Scenario Lattice Generation**: Natural discretization for Markovian processes
- **Conditional Density Estimation**: Generate trajectories from observed data using non-parametric techniques
- **Comprehensive Visualization**: Plot trees, lattices, and probability distributions
- **Well-tested**: Extensive test suite ported from the original Julia package

## Installation

### From Source

```bash
git clone <repository-url>
cd scentrees/python
pip install -e .
```

### Dependencies

The package requires Python 3.10+ and the following packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- openpyxl >= 3.1.0

## Quick Start

### Example 1: Gaussian Random Walk Scenario Tree

```python
from scentrees import Tree, tree_approximation, gaussian_path1d, tree_plot

# Create initial tree structure with branching [1, 2, 2, 2]
tree = Tree.from_branching([1, 2, 2, 2], dimension=1)

# Approximate with 100,000 sample paths
approximated_tree = tree_approximation(tree, gaussian_path1d, 100000, p=2, r=2)

# Plot the result
tree_plot(approximated_tree)
```

### Example 2: Running Maximum Scenario Lattice

```python
from scentrees import lattice_approximation, running_maximum1d, plot_lattice

# Generate lattice with branching structure [1, 2, 3, 4]
lattice = lattice_approximation([1, 2, 3, 4], running_maximum1d, 100000, r=2, dimension=1)

# Plot the lattice
plot_lattice(lattice)
```

### Example 3: Kernel Density Estimation from Data

```python
import numpy as np
from scentrees import gaussian_path1d, kernel_scenarios, Tree, tree_approximation

# Generate sample data (1000 trajectories)
data = np.array([gaussian_path1d()[:, 0] for _ in range(1000)])

# Create trajectory generator from data
generator = kernel_scenarios(data, markovian=False)

# Use generator to create scenario tree
tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
kernel_tree = tree_approximation(tree, generator, 100000, p=2, r=2)
```

## Core Concepts

### Scenario Trees vs Scenario Lattices

- **Scenario Trees**: Used for general discrete-time stochastic processes. Each node has specific children, creating a tree structure where paths branch and don't recombine.

- **Scenario Lattices**: Used for Markovian processes. Nodes at each stage can transition to any node in the next stage, allowing paths to recombine. This is more efficient for Markov processes.

### Stochastic Approximation

The package uses stochastic approximation to iteratively improve the tree/lattice approximation:

1. Start with an initial tree/lattice structure
2. Generate sample paths from the stochastic process
3. Find the closest tree path to each sample
4. Update node states and probabilities using stochastic gradient descent
5. Repeat for many iterations to converge to optimal approximation

## API Reference

### Data Structures

- **`Tree`**: Scenario tree data structure
  - `Tree.from_branching(branching, dimension)`: Create tree from branching structure
  - `Tree.from_identifier(id)`: Create predefined example tree

- **`Lattice`**: Scenario lattice data structure

### Main Functions

- **`tree_approximation(tree, path, n_iterations, p=2, r=2)`**: Approximate stochastic process with scenario tree
- **`lattice_approximation(branching, path, n_iterations, r=2, dimension=1)`**: Approximate Markov process with scenario lattice

### Helper Functions

- `stage(tree, node=None)`: Get stage of nodes
- `height(tree)`: Get tree height
- `leaves(tree, node=None)`: Get leaf nodes and probabilities
- `nodes(tree, t=None)`: Get nodes at specific stage
- `root(tree, node=None)`: Get root or path to node

### Stochastic Process Generators

- `gaussian_path1d()`: 1D Gaussian random walk
- `gaussian_path2d()`: 2D Gaussian random walk
- `running_maximum1d()`: 1D running maximum process
- `running_maximum2d()`: 2D running maximum process
- `path()`: Simple stock price process

### Visualization

- `tree_plot(tree)`: Plot scenario tree with probability density
- `plot_hd(tree)`: Plot multi-dimensional tree
- `plot_lattice(lattice)`: Plot scenario lattice
- `bushiness_nesdistance()`: Plot bushiness vs distance relationship

### Data-Driven Methods

- `kernel_scenarios(data, kernel_distribution=None, markovian=True)`: Generate trajectory generator from data using kernel density estimation

## Advanced Usage

### Custom Stochastic Processes

You can define custom stochastic processes:

```python
import numpy as np

def my_custom_process():
    """Custom stochastic process returning shape (T, d)."""
    T = 4  # number of stages
    d = 1  # dimension
    # Your process logic here
    return np.random.randn(T, d)

tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
approximated = tree_approximation(tree, my_custom_process, 100000)
```

### Multi-dimensional Processes (Now Fully Supported!)

```python
from scentrees import Tree, tree_approximation, gaussian_path2d

# 2D tree
tree_2d = Tree.from_branching([1, 3, 3, 3], dimension=2)
approximated_2d = tree_approximation(tree_2d, gaussian_path2d, 100000)

# Multi-product scenario tree (e.g., 4 correlated products)
import numpy as np
from scentrees import create_multidim_generator

# Load your empirical data: shape (n_scenarios, n_stages, n_products)
# Example: 10,000 scenarios × 6 stages × 4 products
chairs_data = ...  # (10000, 6)
tables_data = ...  # (10000, 6)
desks_data = ...   # (10000, 6)
office_data = ...  # (10000, 6)

all_products_data = np.stack([chairs_data, tables_data, desks_data, office_data], axis=2)

# Create multi-dimensional tree (one tree with all products!)
tree_4d = Tree.from_branching([1, 4, 3, 3], dimension=4)
gen_4d = create_multidim_generator(all_products_data, markovian=False)
approximated_4d = tree_approximation(tree_4d, gen_4d, 150000)

# Each node now has a 4-dimensional state vector [chairs, tables, desks, office]
print(approximated_4d.state.shape)  # (40, 4) - 40 nodes, 4 dimensions
```

### Different Norms and Distance Parameters

```python
# Manhattan distance (p=1) with custom r parameter
tree = Tree.from_branching([1, 2, 2, 2], dimension=1)
approximated = tree_approximation(tree, gaussian_path1d, 100000, p=1, r=2)
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest python/tests/ -v

# Run with coverage
pytest python/tests/ --cov=scentrees --cov-report=html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this package in your research, please cite the original Julia package:

```bibtex
@article{Kirui2020,
    author = {Kirui, Kipngeno and Pichler, Alois and Pflug, Georg Ch.},
    title = {ScenTrees.jl: A Julia Package for Generating Scenario Trees and Scenario Lattices for Multistage Stochastic Programming},
    journal = {Journal of Open Source Software},
    publisher = {The Open Journal},
    year = {2020},
    volume = {5},
    number = {46},
    pages = {1912},
    doi = {10.21105/joss.01912},
    url = {https://doi.org/10.21105/joss.01912}
}
```

## References

- Pflug, Georg Ch., and Alois Pichler. "A Distance for Multistage Stochastic Optimization Models." SIAM Journal on Optimization 22.1 (2012): 1-23. [DOI: 10.1137/110825054](https://doi.org/10.1137/110825054)

- Pflug, Georg Ch., and Alois Pichler. "Dynamic Generation of Scenario Trees." Computational Optimization and Applications 62.3 (2015): 641-668. [DOI: 10.1007/s10589-015-9758-0](https://doi.org/10.1007/s10589-015-9758-0)

- Pflug, Georg Ch., and Alois Pichler. "From Empirical Observations to Tree Models for Stochastic Optimization: Convergence Properties." SIAM Journal on Optimization 26.3 (2016): 1715-1740. [DOI: 10.1137/15M1043376](https://doi.org/10.1137/15M1043376)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original Julia package: [ScenTrees.jl](https://github.com/kirui93/ScenTrees.jl) by Kipngeno Kirui
- Active Julia fork: [ScenTrees.jl](https://github.com/aloispichler/ScenTrees.jl) by Alois Pichler
- Based on theoretical work by Georg Ch. Pflug and Alois Pichler

## Related Packages

- [Original ScenTrees.jl (Julia)](https://github.com/kirui93/ScenTrees.jl)
- [Active fork ScenTrees.jl (Julia)](https://github.com/aloispichler/ScenTrees.jl)