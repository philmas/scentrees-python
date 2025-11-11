"""
Tests for deterministic behavior with set_seed() function.

This module tests that using scentrees.set_seed() produces identical
results across multiple runs for all RNG-dependent operations.
"""

import numpy as np
import pytest
import scentrees


class TestSeedDeterminism:
    """Test deterministic behavior when using set_seed()."""
    
    def test_stochastic_paths_determinism(self):
        """Test that path generators produce identical results with same seed."""
        # First run
        scentrees.set_seed(42)
        path1_1d = scentrees.gaussian_path1d()
        path2_1d = scentrees.gaussian_path2d()
        path3_max = scentrees.running_maximum1d()
        path4_max2d = scentrees.running_maximum2d()
        path5_stock = scentrees.path()
        
        # Second run with same seed
        scentrees.set_seed(42)
        path1_1d_repeat = scentrees.gaussian_path1d()
        path2_1d_repeat = scentrees.gaussian_path2d()
        path3_max_repeat = scentrees.running_maximum1d()
        path4_max2d_repeat = scentrees.running_maximum2d()
        path5_stock_repeat = scentrees.path()
        
        # Verify identical results
        assert np.allclose(path1_1d, path1_1d_repeat), "gaussian_path1d not deterministic"
        assert np.allclose(path2_1d, path2_1d_repeat), "gaussian_path2d not deterministic"
        assert np.allclose(path3_max, path3_max_repeat), "running_maximum1d not deterministic"
        assert np.allclose(path4_max2d, path4_max2d_repeat), "running_maximum2d not deterministic"
        assert np.allclose(path5_stock, path5_stock_repeat), "path not deterministic"
    
    def test_tree_structure_determinism(self):
        """Test that Tree.from_branching produces identical results with same seed."""
        # First run
        scentrees.set_seed(123)
        tree1 = scentrees.Tree.from_branching([1, 2, 2, 2], dimension=1)
        
        # Second run with same seed
        scentrees.set_seed(123)
        tree2 = scentrees.Tree.from_branching([1, 2, 2, 2], dimension=1)
        
        # Verify identical results
        assert np.allclose(tree1.state, tree2.state), "Tree states not deterministic"
        assert np.allclose(tree1.probability, tree2.probability), "Tree probabilities not deterministic"
        assert np.array_equal(tree1.parent, tree2.parent), "Tree structure not deterministic"
    
    def test_tree_structure_multidim_determinism(self):
        """Test that multi-dimensional trees are deterministic."""
        # First run
        scentrees.set_seed(456)
        tree1 = scentrees.Tree.from_branching([1, 3, 3], dimension=4)
        
        # Second run with same seed
        scentrees.set_seed(456)
        tree2 = scentrees.Tree.from_branching([1, 3, 3], dimension=4)
        
        # Verify identical results
        assert tree1.state.shape == tree2.state.shape
        assert np.allclose(tree1.state, tree2.state), "Multi-dim tree states not deterministic"
        assert np.allclose(tree1.probability, tree2.probability), "Multi-dim tree probabilities not deterministic"
    
    def test_kernel_scenarios_determinism(self):
        """Test that kernel_scenarios produces identical trajectories with same seed."""
        # Generate sample data (with seed for consistency)
        scentrees.set_seed(789)
        data = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        
        # First run
        scentrees.set_seed(789)
        data1 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        gen1 = scentrees.kernel_scenarios(data1, markovian=False)
        trajectory1 = gen1()
        
        # Second run with same seed
        scentrees.set_seed(789)
        data2 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        gen2 = scentrees.kernel_scenarios(data2, markovian=False)
        trajectory2 = gen2()
        
        # Verify data is identical
        assert np.allclose(data1, data2), "Generated data not deterministic"
        # Verify trajectories are identical
        assert np.allclose(trajectory1, trajectory2), "kernel_scenarios not deterministic"
    
    def test_kernel_scenarios_markovian_determinism(self):
        """Test that Markovian kernel_scenarios is deterministic."""
        scentrees.set_seed(321)
        data = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        
        # First run
        scentrees.set_seed(321)
        data1 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        gen1 = scentrees.kernel_scenarios(data1, markovian=True)
        trajectory1 = gen1()
        
        # Second run with same seed
        scentrees.set_seed(321)
        data2 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(100)])
        gen2 = scentrees.kernel_scenarios(data2, markovian=True)
        trajectory2 = gen2()
        
        # Verify identical results
        assert np.allclose(data1, data2), "Generated data not deterministic"
        assert np.allclose(trajectory1, trajectory2), "Markovian kernel_scenarios not deterministic"
    
    def test_multidim_generator_determinism(self):
        """Test that create_multidim_generator is deterministic."""
        # Generate sample 3D data
        scentrees.set_seed(654)
        data = np.random.randn(50, 4, 3).cumsum(axis=1)
        
        # First run
        scentrees.set_seed(111)
        gen1 = scentrees.create_multidim_generator(data, markovian=False)
        path1 = gen1()
        
        # Second run with same seed
        scentrees.set_seed(111)
        gen2 = scentrees.create_multidim_generator(data, markovian=False)
        path2 = gen2()
        
        # Verify identical results
        assert np.allclose(path1, path2), "create_multidim_generator not deterministic"
    
    def test_tree_approximation_determinism(self):
        """Test that tree_approximation produces identical results with same seed."""
        # First run
        scentrees.set_seed(999)
        tree1 = scentrees.Tree.from_branching([1, 2, 2, 2], dimension=1)
        approximated1 = scentrees.tree_approximation(tree1, scentrees.gaussian_path1d, 1000, p=2, r=2)
        
        # Second run with same seed
        scentrees.set_seed(999)
        tree2 = scentrees.Tree.from_branching([1, 2, 2, 2], dimension=1)
        approximated2 = scentrees.tree_approximation(tree2, scentrees.gaussian_path1d, 1000, p=2, r=2)
        
        # Verify identical results
        assert np.allclose(approximated1.state, approximated2.state), "tree_approximation states not deterministic"
        assert np.allclose(approximated1.probability, approximated2.probability), "tree_approximation probabilities not deterministic"
    
    def test_lattice_approximation_determinism(self):
        """Test that lattice_approximation produces identical results with same seed."""
        # First run
        scentrees.set_seed(777)
        lattice1 = scentrees.lattice_approximation(
            [1, 2, 3, 4], 
            scentrees.gaussian_path1d, 
            1000, 
            r=2, 
            dimension=1
        )
        
        # Second run with same seed
        scentrees.set_seed(777)
        lattice2 = scentrees.lattice_approximation(
            [1, 2, 3, 4], 
            scentrees.gaussian_path1d, 
            1000, 
            r=2, 
            dimension=1
        )
        
        # Verify identical results
        assert len(lattice1.state) == len(lattice2.state)
        for s1, s2 in zip(lattice1.state, lattice2.state):
            assert np.allclose(s1, s2), "lattice_approximation states not deterministic"
        
        for p1, p2 in zip(lattice1.probability, lattice2.probability):
            assert np.allclose(p1, p2), "lattice_approximation probabilities not deterministic"
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        # First seed
        scentrees.set_seed(111)
        tree1 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
        path1 = scentrees.gaussian_path1d()
        
        # Different seed
        scentrees.set_seed(222)
        tree2 = scentrees.Tree.from_branching([1, 2, 2], dimension=1)
        path2 = scentrees.gaussian_path1d()
        
        # Verify different results
        assert not np.allclose(tree1.state, tree2.state), "Different seeds should produce different trees"
        assert not np.allclose(path1, path2), "Different seeds should produce different paths"
    
    def test_deprecated_set_random_seed_warning(self):
        """Test that set_random_seed raises deprecation warning."""
        with pytest.warns(DeprecationWarning, match="set_random_seed.*deprecated"):
            scentrees.set_random_seed(42)
    
    def test_full_workflow_determinism(self):
        """Test complete workflow from data generation to tree approximation."""
        # Full workflow run 1
        scentrees.set_seed(1234)
        
        # Generate data
        data1 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(50)])
        
        # Create generator
        gen1 = scentrees.kernel_scenarios(data1, markovian=False)
        
        # Create and approximate tree
        tree1 = scentrees.Tree.from_branching([1, 3, 3], dimension=1)
        approximated1 = scentrees.tree_approximation(tree1, gen1, 500)
        
        # Full workflow run 2 with same seed
        scentrees.set_seed(1234)
        
        # Generate data
        data2 = np.array([scentrees.gaussian_path1d()[:, 0] for _ in range(50)])
        
        # Create generator
        gen2 = scentrees.kernel_scenarios(data2, markovian=False)
        
        # Create and approximate tree
        tree2 = scentrees.Tree.from_branching([1, 3, 3], dimension=1)
        approximated2 = scentrees.tree_approximation(tree2, gen2, 500)
        
        # Verify complete workflow is deterministic
        assert np.allclose(data1, data2), "Data generation not deterministic"
        assert np.allclose(tree1.state, tree2.state), "Tree initialization not deterministic"
        assert np.allclose(approximated1.state, approximated2.state), "Full workflow not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])