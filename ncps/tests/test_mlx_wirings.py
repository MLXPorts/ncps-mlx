"""Tests for MLX wiring patterns."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx.wirings import Wiring, FullyConnected, Random, NCP, AutoNCP


def test_base_wiring():
    """Test base Wiring class functionality."""
    units = 10
    wiring = Wiring(units)
    
    # Test initialization
    assert wiring.units == units
    assert wiring.adjacency_matrix.shape == (units, units)
    assert wiring.input_dim is None
    assert wiring.output_dim is None
    
    # Test building
    input_dim = 5
    wiring.build(input_dim)
    assert wiring.input_dim == input_dim
    assert wiring.sensory_adjacency_matrix.shape == (input_dim, units)
    
    # Test synapse addition
    wiring.add_synapse(0, 1, 1)  # Excitatory
    wiring.add_synapse(1, 2, -1)  # Inhibitory
    assert float(wiring.adjacency_matrix[0, 1]) == 1.0
    assert float(wiring.adjacency_matrix[1, 2]) == -1.0
    
    # Test sensory synapse addition
    wiring.add_sensory_synapse(0, 0, 1)
    assert float(wiring.sensory_adjacency_matrix[0, 0]) == 1.0
    
    # Test synapse counting
    assert wiring.synapse_count == 2  # Two internal synapses
    assert wiring.sensory_synapse_count == 1  # One sensory synapse


def test_fully_connected():
    """Test FullyConnected wiring pattern."""
    units = 5
    output_dim = 3
    wiring = FullyConnected(units, output_dim=output_dim, self_connections=True)
    
    # Test initialization
    assert wiring.units == units
    assert wiring.output_dim == output_dim
    
    # Test connectivity
    assert wiring.synapse_count == units * units  # All-to-all connections
    
    # Test building
    input_dim = 4
    wiring.build(input_dim)
    assert wiring.sensory_synapse_count == input_dim * units  # Full sensory connectivity


def test_random_wiring():
    """Test Random wiring pattern."""
    units = 10
    sparsity = 0.5
    wiring = Random(units, sparsity_level=sparsity, random_seed=42)
    
    # Test initialization
    assert wiring.units == units
    assert wiring.sparsity_level == sparsity
    
    # Test connectivity
    expected_synapses = int(np.round(units * units * (1 - sparsity)))
    assert abs(wiring.synapse_count - expected_synapses) <= 1  # Allow for rounding
    
    # Test building
    input_dim = 5
    wiring.build(input_dim)
    expected_sensory_synapses = int(np.round(input_dim * units * (1 - sparsity)))
    assert abs(wiring.sensory_synapse_count - expected_sensory_synapses) <= 1


def test_ncp_wiring():
    """Test Neural Circuit Policy wiring pattern."""
    inter_neurons = 6
    command_neurons = 4
    motor_neurons = 2
    sensory_fanout = 2
    inter_fanout = 2
    recurrent_command_synapses = 2
    motor_fanin = 2
    
    wiring = NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=motor_neurons,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
        seed=42
    )
    
    # Test initialization
    assert wiring.units == inter_neurons + command_neurons + motor_neurons
    assert wiring.output_dim == motor_neurons
    assert wiring.num_layers == 3
    
    # Test neuron types
    assert wiring.get_type_of_neuron(0) == "motor"  # First motor neuron
    assert wiring.get_type_of_neuron(motor_neurons + 1) == "command"  # Command neuron
    assert wiring.get_type_of_neuron(motor_neurons + command_neurons + 1) == "inter"  # Inter neuron
    
    # Test building
    input_dim = 4
    wiring.build(input_dim)
    
    # Test layer connectivity
    motor_layer = wiring.get_neurons_of_layer(2)
    command_layer = wiring.get_neurons_of_layer(1)
    inter_layer = wiring.get_neurons_of_layer(0)
    
    assert len(motor_layer) == motor_neurons
    assert len(command_layer) == command_neurons
    assert len(inter_layer) == inter_neurons


def test_auto_ncp():
    """Test AutoNCP wiring pattern."""
    units = 20
    output_size = 4
    sparsity = 0.5
    
    wiring = AutoNCP(
        units=units,
        output_size=output_size,
        sparsity_level=sparsity,
        seed=42
    )
    
    # Test initialization
    assert wiring.units == units
    assert wiring.output_dim == output_size
    
    # Test building
    input_dim = 6
    wiring.build(input_dim)
    
    # Verify layer structure
    assert len(wiring.get_neurons_of_layer(2)) == output_size  # Motor layer
    assert len(wiring.get_neurons_of_layer(0)) + len(wiring.get_neurons_of_layer(1)) == units - output_size


def test_wiring_errors():
    """Test error handling in wiring patterns."""
    # Test invalid sparsity
    with pytest.raises(ValueError):
        Random(units=10, sparsity_level=1.5)
    
    # Test invalid neuron indices
    wiring = Wiring(units=5)
    with pytest.raises(ValueError):
        wiring.add_synapse(-1, 0, 1)  # Invalid source
    with pytest.raises(ValueError):
        wiring.add_synapse(0, 10, 1)  # Invalid destination
    
    # Test invalid polarity
    with pytest.raises(ValueError):
        wiring.add_synapse(0, 1, 2)  # Invalid polarity
    
    # Test sensory synapse before build
    with pytest.raises(ValueError):
        wiring.add_sensory_synapse(0, 0, 1)
    
    # Test AutoNCP constraints
    with pytest.raises(ValueError):
        AutoNCP(units=10, output_size=9)  # Not enough neurons for layers
    with pytest.raises(ValueError):
        AutoNCP(units=10, output_size=5, sparsity_level=0.0)  # Invalid sparsity


def test_wiring_serialization():
    """Test configuration serialization."""
    # Test FullyConnected
    fc = FullyConnected(units=5, output_dim=3)
    config = fc.get_config()
    fc_restored = FullyConnected.from_config(config)
    assert fc_restored.units == fc.units
    assert fc_restored.output_dim == fc.output_dim
    
    # Test Random
    random = Random(units=8, sparsity_level=0.3)
    config = random.get_config()
    random_restored = Random.from_config(config)
    assert random_restored.units == random.units
    assert random_restored.sparsity_level == random.sparsity_level
    
    # Test AutoNCP
    auto_ncp = AutoNCP(units=15, output_size=3, sparsity_level=0.4)
    config = auto_ncp.get_config()
    auto_ncp_restored = AutoNCP.from_config(config)
    assert auto_ncp_restored.units == auto_ncp.units
    assert auto_ncp_restored._output_size == auto_ncp._output_size


def test_wiring_reproducibility():
    """Test reproducibility with random seeds."""
    # Test Random wiring
    seed = 42
    random1 = Random(units=10, random_seed=seed)
    random2 = Random(units=10, random_seed=seed)
    assert mx.array_equal(random1.adjacency_matrix, random2.adjacency_matrix)
    
    # Test AutoNCP
    auto_ncp1 = AutoNCP(units=20, output_size=4, seed=seed)
    auto_ncp2 = AutoNCP(units=20, output_size=4, seed=seed)
    assert mx.array_equal(auto_ncp1.adjacency_matrix, auto_ncp2.adjacency_matrix)
