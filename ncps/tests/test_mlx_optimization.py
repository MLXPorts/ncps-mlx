"""Tests for wiring optimization techniques."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring


def optimize_sparsity(wiring, target_density=0.1):
    """Optimize wiring sparsity while maintaining performance."""
    # Get initial connectivity
    initial_synapses = wiring.synapse_count
    target_synapses = int(initial_synapses * target_density)
    
    # Get synapse strengths
    strengths = np.abs(wiring.adjacency_matrix)
    
    # Sort synapses by strength
    sorted_indices = np.argsort(strengths.flatten())
    
    # Remove weakest synapses
    to_remove = sorted_indices[:initial_synapses - target_synapses]
    for idx in to_remove:
        i, j = np.unravel_index(idx, strengths.shape)
        wiring.adjacency_matrix = wiring.adjacency_matrix.at[i, j].set(0)
    
    return wiring


def optimize_connectivity(wiring, importance_scores):
    """Optimize connectivity based on importance scores."""
    # Strengthen important connections
    mask = wiring.adjacency_matrix != 0
    wiring.adjacency_matrix = wiring.adjacency_matrix * (1 + importance_scores * mask)
    return wiring


def compute_importance(activations):
    """Compute importance scores from activations."""
    # Simple importance metric based on activation variance
    return np.var(activations, axis=0)


class OptimizedWiring(Wiring):
    """Test wiring class with optimization."""
    
    def __init__(self, units, target_density=0.1):
        super().__init__(units)
        self.target_density = target_density
        
        # Build initial dense connectivity
        for i in range(units):
            for j in range(units):
                if i != j:
                    self.add_synapse(i, j, 1)
        
        # Optimize
        self._optimize()
    
    def _optimize(self):
        """Apply optimizations."""
        # Sparsify
        optimize_sparsity(self, self.target_density)


def test_sparsity_optimization():
    """Test sparsity optimization."""
    # Create dense wiring
    units = 32
    wiring = Wiring(units)
    for i in range(units):
        for j in range(units):
            if i != j:
                wiring.add_synapse(i, j, 1)
    
    # Initial density
    initial_density = wiring.synapse_count / (units * units)
    
    # Target density
    target_density = 0.1
    
    # Optimize
    optimized = optimize_sparsity(wiring, target_density)
    
    # Check density
    final_density = optimized.synapse_count / (units * units)
    assert abs(final_density - target_density) < 0.05  # Allow small deviation


def test_connectivity_optimization():
    """Test connectivity optimization."""
    # Create wiring
    units = 16
    wiring = Wiring(units)
    for i in range(units):
        for j in range(units):
            if i != j:
                wiring.add_synapse(i, j, 1)
    
    # Generate importance scores
    importance = np.random.rand(units, units)
    
    # Initial weights
    initial_weights = mx.array(wiring.adjacency_matrix)
    
    # Optimize
    optimized = optimize_connectivity(wiring, importance)
    
    # Check that important connections are strengthened
    high_importance = importance > np.median(importance)
    assert mx.all(
        mx.abs(optimized.adjacency_matrix[high_importance]) >= 
        mx.abs(initial_weights[high_importance])
    )


def test_optimized_wiring():
    """Test OptimizedWiring class."""
    # Create optimized wiring
    units = 64
    target_density = 0.2
    wiring = OptimizedWiring(units, target_density)
    
    # Check density
    density = wiring.synapse_count / (units * units)
    assert abs(density - target_density) < 0.05
    
    # Create model
    model = CfC(wiring=wiring)
    
    # Test forward pass
    batch_size = 16
    seq_length = 10
    input_dim = 8
    x = mx.random.normal((batch_size, seq_length, input_dim))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, wiring.output_dim)


def test_optimization_with_training():
    """Test optimization during training."""
    # Create wiring
    units = 32
    wiring = OptimizedWiring(units, target_density=0.3)
    
    # Create model
    model = LTC(wiring=wiring)
    
    # Generate data
    batch_size = 8
    seq_length = 15
    input_dim = 4
    x = mx.random.normal((batch_size, seq_length, input_dim))
    y = mx.random.normal((batch_size, seq_length, wiring.output_dim))
    
    # Training step
    optimizer = mx.optimizers.Adam(learning_rate=0.001)
    
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    
    assert not mx.isnan(loss)
    assert float(loss) > 0


def test_importance_computation():
    """Test importance score computation."""
    # Generate activations
    units = 16
    samples = 100
    activations = np.random.randn(samples, units, units)
    
    # Compute importance
    importance = compute_importance(activations)
    
    # Check shape
    assert importance.shape == (units, units)
    
    # Check values
    assert np.all(importance >= 0)  # Variance is non-negative


def test_optimization_errors():
    """Test error handling in optimization."""
    # Test invalid density
    with pytest.raises(ValueError):
        OptimizedWiring(units=10, target_density=1.5)
    
    # Test invalid importance scores
    wiring = Wiring(units=8)
    with pytest.raises(ValueError):
        optimize_connectivity(wiring, np.random.rand(10, 10))  # Wrong shape


def test_optimization_reproducibility():
    """Test optimization reproducibility."""
    # Create two wirings with same seed
    seed = 42
    np.random.seed(seed)
    wiring1 = OptimizedWiring(units=16, target_density=0.2)
    
    np.random.seed(seed)
    wiring2 = OptimizedWiring(units=16, target_density=0.2)
    
    # Check that optimizations are identical
    assert mx.array_equal(wiring1.adjacency_matrix, wiring2.adjacency_matrix)


def test_optimization_stability():
    """Test optimization stability."""
    # Create wiring
    wiring = OptimizedWiring(units=32, target_density=0.2)
    
    # Record initial state
    initial_density = wiring.synapse_count / (32 * 32)
    initial_matrix = mx.array(wiring.adjacency_matrix)
    
    # Apply optimization multiple times
    for _ in range(5):
        optimize_sparsity(wiring, target_density=0.2)
    
    # Check stability
    final_density = wiring.synapse_count / (32 * 32)
    assert abs(final_density - initial_density) < 0.05
    
    # Major structure should be preserved
    correlation = mx.mean(mx.sign(wiring.adjacency_matrix) == mx.sign(initial_matrix))
    assert float(correlation) > 0.8  # At least 80% of signs preserved
