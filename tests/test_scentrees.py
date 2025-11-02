"""
Test suite for scentrees package.

Ports tests from the Julia ScenTrees.jl test suite to Python using pytest.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Import package functions
from scentrees import (
    Tree, Lattice,
    tree_approximation, lattice_approximation,
    stage, height, leaves, nodes, root,
    gaussian_path1d, gaussian_path2d,
    running_maximum1d, running_maximum2d, path,
    kernel_scenarios
)

# Test data directory
TEST_DATA_DIR = Path(__file__).parent


class TestPredefinedTrees:
    """Tests for predefined tree structures."""
    
    def test_tree_402(self):
        """Test predefined tree 402."""
        a = Tree.from_identifier(402)
        assert isinstance(a, Tree)
        assert len(a.parent) == 15
        assert len(a.state) == len(a.probability) == len(a.parent) == 15
        assert np.isclose(np.sum(a.probability), 8.0)
        assert len(a.children) == 8
        assert len(root(a)) == 1


class TestInitialTrees:
    """Tests for initial tree generation."""
    
    def test_initial_tree(self):
        """Test creating a tree from branching structure."""
        init = Tree.from_branching([1, 2, 2, 2], dimension=1)
        assert isinstance(init, Tree)
        assert len(init.parent) == 15
        assert len(init.state) == len(init.probability) == len(init.parent) == 15
        assert len(init.children) == 8
        assert len(stage(init)) == 15
        assert height(init) == 3
        assert len(leaves(init)[0]) == 8  # 2^3 leaves
        assert len(nodes(init)) == 15
        assert len(root(init)) == 1


class TestScenarioTrees:
    """Tests for scenario tree generation."""
    
    def test_tree_1d(self):
        """Test 1D scenario tree."""
        x = Tree.from_branching([1, 3, 3, 3, 3], dimension=1)
        assert isinstance(x, Tree)
        assert len(x.parent) == 121
        assert len(x.state) == len(x.probability) == len(x.parent) == 121
        assert np.isclose(np.sum(x.probability), 41.0)
        assert len(x.children) == 41
        assert len(root(x)) == 1
        assert len(leaves(x)[0]) == 81  # 3^4
    
    def test_tree_2d(self):
        """Test 2D scenario tree."""
        y = Tree.from_branching([1, 3, 3, 3, 3], dimension=2)
        assert isinstance(y, Tree)
        assert len(y.parent) == 121
        assert len(y.probability) == len(y.parent) == 121
        assert y.state.shape == (121, 2)
        assert np.isclose(np.sum(y.probability), 41.0)
        assert len(y.children) == 41


class TestStochasticFunctions:
    """Tests for stochastic path generators."""
    
    def test_gaussian_path1d(self):
        """Test 1D Gaussian path."""
        a = gaussian_path1d()
        assert a.shape == (4, 1)
        assert a[0, 0] == 0.0  # Starts at 0
    
    def test_running_maximum1d(self):
        """Test 1D running maximum."""
        b = running_maximum1d()
        assert b.shape == (4, 1)
        # Each value should be >= previous (running maximum property)
        for i in range(1, 4):
            assert b[i, 0] >= b[i-1, 0]
    
    def test_path(self):
        """Test stock price path."""
        c = path()
        assert c.shape == (4, 1)
        assert c[0, 0] == 100.0  # Starts at 100
    
    def test_gaussian_path2d(self):
        """Test 2D Gaussian path."""
        d = gaussian_path2d()
        assert d.shape == (4, 2)
    
    def test_running_maximum2d(self):
        """Test 2D running maximum."""
        e = running_maximum2d()
        assert e.shape == (4, 2)


class TestTreeApproximation:
    """Tests for tree approximation algorithm."""
    
    @pytest.mark.parametrize("path_func", [gaussian_path1d, running_maximum1d])
    @pytest.mark.parametrize("branching", [[1, 2, 2, 2], [1, 3, 3, 3]])
    def test_tree_approximation_1d(self, path_func, branching):
        """Test 1D tree approximation with different paths and branching."""
        newtree = Tree.from_branching(branching, dimension=1)
        sample_size = 10000  # Reduced for faster testing
        p = 2
        r = 2
        
        tree_approximation(newtree, path_func, sample_size, p, r)
        
        assert len(newtree.parent) == len(newtree.state[:, 0])
        assert len(newtree.parent) == len(newtree.probability)
        assert len(stage(newtree)) == len(newtree.parent)
        assert height(newtree) == np.max(stage(newtree))
        
        # Sum of leaf probabilities should be approximately 1
        leaf_probs = leaves(newtree)[2]
        assert np.isclose(np.sum(leaf_probs), 1.0, atol=0.1)
        assert len(root(newtree)) == 1
    
    def test_tree_approximation_2d(self):
        """Test 2D tree approximation."""
        twoD = Tree.from_branching([1, 3, 3, 3], dimension=2)
        tree_approximation(twoD, gaussian_path2d, 10000, 2, 2)
        
        assert twoD.state.shape[1] == 2
        assert twoD.state.shape[0] == len(twoD.parent) == len(twoD.probability)


class TestLatticeApproximation:
    """Tests for lattice approximation algorithm."""
    
    def test_lattice_approximation_1d(self):
        """Test 1D lattice approximation."""
        tstLat = lattice_approximation([1, 2, 3, 4], gaussian_path1d, 50000, r=2, dimension=1)
        
        assert len(tstLat.state) == len(tstLat.probability)
        
        # Sum of probabilities at each stage should be approximately 1
        prob_sums = [np.round(np.sum(p), decimals=1) for p in tstLat.probability]
        assert prob_sums == [1.0, 1.0, 1.0, 1.0]
    
    def test_lattice_approximation_2d(self):
        """Test 2D lattice approximation."""
        lat2 = lattice_approximation([1, 2, 3, 4], gaussian_path2d, 50000, r=2, dimension=2)
        
        assert len(lat2) == 2  # Two lattices (one per dimension)
        assert len(lat2[0].state) == len(lat2[0].probability)
        
        # Check probabilities for both dimensions
        prob_sums_1 = [np.round(np.sum(p), decimals=1) for p in lat2[0].probability]
        prob_sums_2 = [np.round(np.sum(p), decimals=1) for p in lat2[1].probability]
        assert prob_sums_1 == [1.0, 1.0, 1.0, 1.0]
        assert prob_sums_2 == [1.0, 1.0, 1.0, 1.0]


class TestExampleData:
    """Tests using example data files."""
    
    def test_csv_data(self):
        """Test loading CSV data."""
        data_file = TEST_DATA_DIR / "data5.csv"
        if not data_file.exists():
            pytest.skip("Test data file not found")
        
        data = pd.read_csv(data_file)
        gsData = data.values
        assert gsData.shape == (100, 10)
    
    def test_excel_data(self):
        """Test loading Excel data."""
        excel_file = TEST_DATA_DIR / "Mappe1.xlsx"
        if not excel_file.exists():
            pytest.skip("Test data file not found")
        
        df1 = pd.read_excel(excel_file, sheet_name="Sheet1")
        df1_array = df1.values.astype(float)
        assert df1_array.shape == (1000, 7)
        
        df2 = pd.read_excel(excel_file, sheet_name="Sheet2")
        df2_array = df2.values.astype(float)
        assert df2_array.shape == (1000, 4)
    
    def test_random_walk_data(self):
        """Test random walk data and kernel density estimation."""
        rw_file = TEST_DATA_DIR / "RandomDataWalk.csv"
        if not rw_file.exists():
            pytest.skip("Test data file not found")
        
        RandomWalkData = pd.read_csv(rw_file)
        RWData = RandomWalkData.values
        assert RWData.shape == (1000, 5)
        
        # Test standard deviations
        sd = np.std(RWData, axis=0)
        assert np.all(sd < 5)
        
        # Test kernel density lattice generation (smaller sample for speed)
        LatFromKernel = lattice_approximation(
            [1, 3, 4, 5, 6],
            kernel_scenarios(RWData),
            10000,
            r=2,
            dimension=1
        )
        
        prob_sums = [np.round(np.sum(p), decimals=1) for p in LatFromKernel.probability]
        assert prob_sums == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert len(LatFromKernel.state) == len(LatFromKernel.probability)


class TestKernelDensity:
    """Tests for kernel density estimation."""
    
    def test_kernel_scenarios_markovian(self):
        """Test Markovian trajectory generation."""
        # Generate sample data
        data = np.array([gaussian_path1d()[:, 0] for _ in range(100)])
        
        # Create generator
        gen = kernel_scenarios(data, markovian=True)
        
        # Generate trajectories
        traj = gen()
        assert traj.shape == (data.shape[1], 1)
    
    def test_kernel_scenarios_non_markovian(self):
        """Test non-Markovian trajectory generation."""
        # Generate sample data
        data = np.array([gaussian_path1d()[:, 0] for _ in range(100)])
        
        # Create generator
        gen = kernel_scenarios(data, markovian=False)
        
        # Generate trajectories
        traj = gen()
        assert traj.shape == (data.shape[1], 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])