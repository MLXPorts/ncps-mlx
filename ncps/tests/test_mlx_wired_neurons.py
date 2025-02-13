"""Tests for wired liquid neurons."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import FullyConnected, Random, NCP, AutoNCP


def test_ltc_default_wiring():
    """Test LTC with default fully connected wiring."""
    # Create wiring
    input_size = 8
    hidden_size = 32
    wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
    wiring.build(input_size)
    
    # Create model
    model = LTC(
        wiring=wiring,
        return_sequences=True,
        activation="tanh"
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, input_size))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, hidden_size)
    
    # Test with time_delta
    time_delta = mx.ones((batch_size, seq_length))
    output_with_time = model(x, time_delta=time_delta)
    assert output_with_time.shape == (batch_size, seq_length, hidden_size)


def test_cfc_modes():
    """Test CfC with different operation modes."""
    input_size = 8
    hidden_size = 32
    
    # Create wiring
    wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
    wiring.build(input_size)
    
    # Test pure mode
    model_pure = CfC(
        wiring=wiring,
        mode="pure",
        return_sequences=True
    )
    
    # Test no_gate mode
    model_no_gate = CfC(
        wiring=wiring,
        mode="no_gate",
        return_sequences=True
    )
    
    # Create test data
    batch_size = 8
    seq_length = 15
    x = mx.random.normal((batch_size, seq_length, input_size))
    
    # Test forward passes
    output_pure = model_pure(x)
    assert output_pure.shape == (batch_size, seq_length, hidden_size)
    
    output_no_gate = model_no_gate(x)
    assert output_no_gate.shape == (batch_size, seq_length, hidden_size)


def test_wiring_validation():
    """Test comprehensive wiring validation."""
    # Test NCP wiring constraints
    with pytest.raises(ValueError):
        # Invalid: motor_fanin > command_neurons
        NCP(
            inter_neurons=8,
            command_neurons=4,
            motor_neurons=2,
            sensory_fanout=2,
            inter_fanout=2,
            recurrent_command_synapses=1,
            motor_fanin=5  # Invalid: greater than command_neurons
        )
    
    with pytest.raises(ValueError):
        # Invalid: inter_fanout > command_neurons
        NCP(
            inter_neurons=8,
            command_neurons=4,
            motor_neurons=2,
            sensory_fanout=2,
            inter_fanout=5,  # Invalid: greater than command_neurons
            recurrent_command_synapses=1,
            motor_fanin=2
        )
    
    # Test Random wiring constraints
    with pytest.raises(ValueError):
        Random(units=16, output_dim=4, sparsity_level=1.0)  # Invalid sparsity
    
    with pytest.raises(ValueError):
        Random(units=16, output_dim=4, sparsity_level=-0.1)  # Invalid sparsity


def test_state_consistency():
    """Test state consistency across time steps."""
    # Create model
    wiring = FullyConnected(units=16, output_dim=4)
    wiring.build(8)  # Build with input size
    model = CfC(wiring=wiring, return_state=True)
    
    # Create data
    batch_size = 4
    seq_length = 10
    input_dim = 8  # Match wiring input dimension
    x = mx.random.normal((batch_size, seq_length, input_dim))
    
    # Initial forward pass
    initial_states = [mx.zeros((batch_size, 16))]
    output1, state1 = model(x, initial_states=initial_states)
    
    # Second forward pass with same input
    output2, state2 = model(x, initial_states=initial_states)
    
    # Check consistency
    assert mx.array_equal(output1, output2)
    assert mx.array_equal(state1[0], state2[0])
    
    # Test state evolution
    x_single = mx.random.normal((batch_size, 1, input_dim))
    current_states = initial_states
    
    states = []
    for _ in range(5):
        _, current_states = model(x_single, initial_states=current_states)
        states.append(current_states[0])
    
    # Verify states are different (evolution occurs)
    for i in range(len(states)-1):
        assert not mx.array_equal(states[i], states[i+1])


def test_cfc_with_fully_connected():
    """Test CfC with fully connected wiring."""
    # Create wiring
    input_dim = 8
    wiring = FullyConnected(units=32, output_dim=10)
    wiring.build(input_dim)
    
    # Create model
    model = CfC(
        wiring=wiring,
        activation="tanh",
        backbone_units=[64],
        backbone_layers=1
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, input_dim))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 10)


def test_ltc_with_random_wiring():
    """Test LTC with random sparse wiring."""
    # Create wiring
    input_dim = 8
    wiring = Random(
        units=32,
        output_dim=10,
        sparsity_level=0.5
    )
    wiring.build(input_dim)
    
    # Create model
    model = LTC(
        wiring=wiring,
        activation="tanh"
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, input_dim))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 10)


def test_cfc_with_ncp():
    """Test CfC with NCP wiring."""
    # Create wiring
    input_dim = 8
    wiring = NCP(
        inter_neurons=16,
        command_neurons=8,
        motor_neurons=4,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=3,
        motor_fanin=4
    )
    wiring.build(input_dim)
    
    # Create model
    model = CfC(
        wiring=wiring,
        activation="tanh"
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, input_dim))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 4)
    
    # Check neuron types
    assert wiring.get_type_of_neuron(0) == "motor"
    assert wiring.get_type_of_neuron(5) == "command"
    assert wiring.get_type_of_neuron(25) == "inter"


def test_ltc_with_auto_ncp():
    """Test LTC with AutoNCP wiring."""
    # Create wiring
    input_dim = 8
    wiring = AutoNCP(
        units=32,
        output_size=4,
        sparsity_level=0.5
    )
    wiring.build(input_dim)
    
    # Create model
    model = LTC(
        wiring=wiring,
        activation="tanh"
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, input_dim))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 4)


def test_wired_model_training():
    """Test training of wired models."""
    # Create wiring and model
    input_dim = 8
    wiring = FullyConnected(units=16, output_dim=1)
    wiring.build(input_dim)  # Build with input size
    model = CfC(wiring=wiring)
    
    # Create data
    batch_size = 32
    seq_length = 10
    x = mx.random.normal((batch_size, seq_length, input_dim))
    y = mx.random.normal((batch_size, seq_length, 1))
    
    def loss_fn(params, x, y):
        model.update(params)
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    loss_and_grad = mx.value_and_grad(loss_fn)
    
    # Single training step
    loss, grads = loss_and_grad(model.parameters(), x, y)
    
    assert not mx.isnan(loss)
    assert float(loss) > 0


def test_wired_model_with_time():
    """Test wired models with time-aware processing."""
    # Create wiring and model
    input_dim = 8
    wiring = Random(units=16, output_dim=1, sparsity_level=0.3)
    wiring.build(input_dim)  # Build with input size
    model = LTC(wiring=wiring)
    
    # Create data with time_delta
    batch_size = 8
    seq_length = 15
    x = mx.random.normal((batch_size, seq_length, input_dim))
    time_delta = mx.ones((batch_size, seq_length))
    
    # Forward pass with time_delta
    output = model(x, time_delta=time_delta)
    assert output.shape == (batch_size, seq_length, 1)
    
    # Forward pass with variable time_delta
    time_delta = mx.random.uniform(0.5, 1.5, (batch_size, seq_length))
    output = model(x, time_delta=time_delta)
    assert output.shape == (batch_size, seq_length, 1)


def test_wired_model_state():
    """Test state handling in wired models."""
    # Create wiring and model
    input_dim = 8
    wiring = FullyConnected(units=16, output_dim=2)
    wiring.build(input_dim)  # Build with input size
    model = CfC(wiring=wiring, return_state=True)
    
    # Create data
    batch_size = 4
    seq_length = 10
    x = mx.random.normal((batch_size, seq_length, input_dim))
    
    # Test with initial states
    initial_states = [mx.zeros((batch_size, 16))]
    output, final_states = model(x, initial_states=initial_states)
    
    assert output.shape == (batch_size, seq_length, 2)
    assert final_states[0].shape == (batch_size, 16)
    
    # Test state propagation
    new_output, new_states = model(x, initial_states=final_states)
    assert new_states[0].shape == final_states[0].shape


def test_wired_model_bidirectional():
    """Test bidirectional processing with wired models."""
    # Create wiring and model
    input_dim = 8
    wiring = Random(units=16, output_dim=2, sparsity_level=0.4)
    wiring.build(input_dim)  # Build with input size
    model = LTC(
        wiring=wiring,
        bidirectional=True
    )
    
    # Create data
    batch_size = 4
    seq_length = 10
    x = mx.random.normal((batch_size, seq_length, input_dim))
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, seq_length, 4)  # Double output size due to concat


def test_wired_model_backbone():
    """Test wired models with backbone networks."""
    # Create wiring and model
    input_dim = 8
    wiring = AutoNCP(units=16, output_size=2)
    wiring.build(input_dim)  # Build with input size
    model = CfC(
        wiring=wiring,
        backbone_units=[32, 16],
        backbone_layers=2,
        backbone_dropout=0.1
    )
    
    # Create data
    batch_size = 4
    seq_length = 10
    x = mx.random.normal((batch_size, seq_length, input_dim))  # Match wiring input dimension
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, seq_length, 2)


def test_wiring_connectivity_patterns():
    """Test wiring connectivity patterns and synapse counts."""
    input_dim = 8
    hidden_dim = 32
    output_dim = 4
    
    # Test fully connected connectivity
    wiring_full = FullyConnected(units=hidden_dim, output_dim=output_dim)
    wiring_full.build(input_dim)
    assert wiring_full.synapse_count == hidden_dim * hidden_dim  # All-to-all connections
    assert wiring_full.sensory_synapse_count == input_dim * hidden_dim
    
    # Test random sparse connectivity
    sparsity = 0.5
    wiring_sparse = Random(units=hidden_dim, output_dim=output_dim, sparsity_level=sparsity)
    wiring_sparse.build(input_dim)
    expected_synapses = int(hidden_dim * hidden_dim * (1 - sparsity))
    assert abs(wiring_sparse.synapse_count - expected_synapses) <= hidden_dim  # Allow small deviation
    
    # Test NCP hierarchical connectivity
    wiring_ncp = NCP(
        inter_neurons=16,
        command_neurons=8,
        motor_neurons=4,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=3,
        motor_fanin=4
    )
    wiring_ncp.build(input_dim)
    
    # Verify NCP connectivity structure
    for i in range(28):  # Total neurons = 16 + 8 + 4 = 28
        neuron_type = wiring_ncp.get_type_of_neuron(i)
        if i < 4:
            assert neuron_type == "motor"
        elif i < 12:
            assert neuron_type == "command"
        else:
            assert neuron_type == "inter"
    
    # Test AutoNCP connectivity
    wiring_auto = AutoNCP(units=32, output_size=4, sparsity_level=0.5)
    wiring_auto.build(input_dim)
    assert wiring_auto.output_dim == 4
    
    # Verify synapse count is within expected range for sparsity
    total_possible = 32 * 32
    assert wiring_auto.synapse_count < total_possible  # Should be sparse
    assert wiring_auto.synapse_count > 0  # Should have some connections
