"""Tests for edge cases and potential issues in wiring patterns."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring, FullyConnected, Random, NCP, AutoNCP


def test_memory_management():
    """Test memory management with MLX arrays."""
    # Test array conversion and updates
    wiring = Wiring(units=10)
    
    # Initial state should be MLX array
    assert isinstance(wiring.adjacency_matrix, mx.array)
    
    # Test multiple updates
    for i in range(5):
        wiring.add_synapse(i, (i + 1) % 10, 1)
        # Should still be MLX array
        assert isinstance(wiring.adjacency_matrix, mx.array)
    
    # Test sensory matrix
    wiring.build(input_dim=5)
    assert isinstance(wiring.sensory_adjacency_matrix, mx.array)
    
    # Test multiple sensory updates
    for i in range(5):
        wiring.add_sensory_synapse(i, i, 1)
        assert isinstance(wiring.sensory_adjacency_matrix, mx.array)


def test_numerical_stability():
    """Test numerical stability in wiring patterns."""
    # Test with very large number of neurons
    large_wiring = Random(
        units=1000,
        sparsity_level=0.99  # Very sparse
    )
    
    # Check that sparsity is maintained
    density = large_wiring.synapse_count / (1000 * 1000)
    assert abs(density - 0.01) < 0.005  # Should be close to 1% density
    
    # Test with very small weights
    small_weights = mx.array(large_wiring.adjacency_matrix) * 1e-6
    large_wiring.adjacency_matrix = small_weights
    
    # Create model
    model = CfC(wiring=large_wiring)
    
    # Test forward pass
    x = mx.random.normal((1, 10, 8))
    output = model(x)
    
    # Check for NaN/Inf
    assert not mx.any(mx.isnan(output))
    assert not mx.any(mx.isinf(output))


def test_connectivity_validation():
    """Test edge cases in connectivity validation."""
    wiring = Wiring(units=5)
    
    # Test self-loops
    wiring.add_synapse(0, 0, 1)  # Should work
    
    # Test boundary conditions
    with pytest.raises(ValueError):
        wiring.add_synapse(-1, 0, 1)  # Invalid source
    
    with pytest.raises(ValueError):
        wiring.add_synapse(0, 5, 1)  # Invalid destination
    
    with pytest.raises(ValueError):
        wiring.add_synapse(0, 0, 0)  # Invalid polarity
    
    # Test sensory boundary conditions
    wiring.build(input_dim=3)
    
    with pytest.raises(ValueError):
        wiring.add_sensory_synapse(3, 0, 1)  # Invalid source
    
    with pytest.raises(ValueError):
        wiring.add_sensory_synapse(0, 5, 1)  # Invalid destination


def test_serialization_edge_cases():
    """Test serialization with edge cases."""
    # Test empty wiring
    wiring = Wiring(units=5)
    config = wiring.get_config()
    restored = Wiring.from_config(config)
    assert mx.array_equal(wiring.adjacency_matrix, restored.adjacency_matrix)
    
    # Test with no sensory connections
    wiring.build(input_dim=3)
    config = wiring.get_config()
    restored = Wiring.from_config(config)
    assert mx.array_equal(wiring.sensory_adjacency_matrix, restored.sensory_adjacency_matrix)
    
    # Test with mixed polarities
    wiring.add_synapse(0, 1, 1)
    wiring.add_synapse(1, 2, -1)
    config = wiring.get_config()
    restored = Wiring.from_config(config)
    assert mx.array_equal(wiring.adjacency_matrix, restored.adjacency_matrix)


def test_random_seed_consistency():
    """Test random seed consistency across platforms."""
    seed = 12345
    
    # Create multiple random wirings with same seed
    wirings = [Random(units=20, random_seed=seed) for _ in range(3)]
    
    # All should have identical connectivity
    for w1, w2 in zip(wirings[:-1], wirings[1:]):
        assert mx.array_equal(w1.adjacency_matrix, w2.adjacency_matrix)
    
    # Test NCP with same seed
    ncps = [
        NCP(
            inter_neurons=10,
            command_neurons=5,
            motor_neurons=3,
            sensory_fanout=2,
            inter_fanout=2,
            recurrent_command_synapses=2,
            motor_fanin=2,
            seed=seed
        )
        for _ in range(3)
    ]
    
    # All should have identical connectivity
    for n1, n2 in zip(ncps[:-1], ncps[1:]):
        assert mx.array_equal(n1.adjacency_matrix, n2.adjacency_matrix)


def test_gradient_flow():
    """Test gradient flow through wiring patterns."""
    # Create wiring with known path
    wiring = Wiring(units=3)
    wiring.add_synapse(0, 1, 1)
    wiring.add_synapse(1, 2, 1)
    
    # Create model
    model = CfC(wiring=wiring)
    
    # Forward and backward pass
    x = mx.random.normal((1, 5, 2))
    y = mx.random.normal((1, 5, 1))
    
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
    
    # Check that gradients exist
    assert len(grads) > 0
    for g in grads.values():
        assert not mx.all(g == 0)


def test_large_scale_stability():
    """Test stability with large-scale networks."""
    # Create large NCP
    wiring = AutoNCP(
        units=1000,
        output_size=100,
        sparsity_level=0.9  # Very sparse
    )
    
    # Verify neuron counts
    assert len(wiring.get_neurons_of_layer(0)) > 0  # Inter neurons
    assert len(wiring.get_neurons_of_layer(1)) > 0  # Command neurons
    assert len(wiring.get_neurons_of_layer(2)) == 100  # Motor neurons
    
    # Test connectivity
    assert wiring.synapse_count > 0
    density = wiring.synapse_count / (1000 * 1000)
    assert density < 0.2  # Should be sparse
    
    # Create model
    model = LTC(wiring=wiring)
    
    # Test forward pass
    x = mx.random.normal((1, 5, 8))
    output = model(x)
    
    # Check output
    assert output.shape == (1, 5, 100)
    assert not mx.any(mx.isnan(output))
    assert not mx.any(mx.isinf(output))


def test_connectivity_patterns():
    """Test specific connectivity patterns."""
    # Test chain connectivity
    wiring = Wiring(units=5)
    for i in range(4):
        wiring.add_synapse(i, i+1, 1)
    
    # Should have exactly 4 connections
    assert wiring.synapse_count == 4
    
    # Test skip connections
    wiring = Wiring(units=5)
    for i in range(3):
        wiring.add_synapse(i, i+2, 1)
    
    # Should have exactly 3 connections
    assert wiring.synapse_count == 3
    
    # Test bidirectional connections
    wiring = Wiring(units=5)
    for i in range(4):
        wiring.add_synapse(i, i+1, 1)
        wiring.add_synapse(i+1, i, -1)
    
    # Should have exactly 8 connections
    assert wiring.synapse_count == 8


def test_type_stability():
    """Test type stability across operations."""
    wiring = Random(units=10, sparsity_level=0.5)
    
    # Initial type check
    assert wiring.adjacency_matrix.dtype == mx.float32
    
    # After operations
    wiring.add_synapse(0, 1, 1)
    assert wiring.adjacency_matrix.dtype == mx.float32
    
    # After build
    wiring.build(input_dim=5)
    assert wiring.sensory_adjacency_matrix.dtype == mx.float32
    
    # After serialization
    config = wiring.get_config()
    restored = Random.from_config(config)
    assert restored.adjacency_matrix.dtype == mx.float32


def test_backward_compatibility():
    """Test backward compatibility with saved configs."""
    # Old-style config (without some newer fields)
    old_config = {
        "units": 5,
        "adjacency_matrix": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        "input_dim": 2,
        "output_dim": 1
    }
    
    # Should still load correctly
    wiring = Wiring.from_config(old_config)
    assert wiring.units == 5
    assert wiring.input_dim == 2
    assert wiring.output_dim == 1
