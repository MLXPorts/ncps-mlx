"""Tests for Keras neural circuit implementations."""

import numpy as np
import pytest
import keras
import json

from ncps.keras import (
    CfC,
    LTC,
    FullyConnected,
    Random,
    NCP,
    AutoNCP,
    CfCCell,
    LTCCell,
    lecun_tanh,
)


def test_cfc_default_wiring():
    """Test CfC with default fully connected wiring."""
    # Create wiring
    input_size = 8
    hidden_size = 32
    wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
    wiring.build(input_size)
    
    # Create model
    model = CfC(
        wiring=wiring,
        return_sequences=True,
        activation="tanh"
    )
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = np.random.normal(size=(batch_size, seq_length, input_size)).astype(np.float32)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, hidden_size)


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
    x = np.random.normal(size=(batch_size, seq_length, input_size)).astype(np.float32)
    
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
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    
    # Initial forward pass
    initial_states = [np.zeros((batch_size, 16), dtype=np.float32)]
    output1, state1 = model(x, initial_state=initial_states)
    
    # Second forward pass with same input
    output2, state2 = model(x, initial_state=initial_states)
    
    # Check consistency
    np.testing.assert_allclose(output1, output2, rtol=1e-5)
    np.testing.assert_allclose(state1[0], state2[0], rtol=1e-5)


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
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
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
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 4)


def test_model_training():
    """Test training of wired models."""
    # Create wiring and model
    input_dim = 8
    wiring = FullyConnected(units=16, output_dim=1)
    wiring.build(input_dim)  # Build with input size
    model = CfC(wiring=wiring)
    
    # Create data
    batch_size = 32
    seq_length = 10
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    y = np.random.normal(size=(batch_size, seq_length, 1)).astype(np.float32)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Single training step
    history = model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    
    assert not np.isnan(loss)
    assert float(loss) > 0


def test_model_serialization():
    """Test model serialization."""
    # Create original model
    wiring = AutoNCP(units=32, output_size=4, sparsity_level=0.5)
    wiring.build(8)
    
    original_model = CfC(
        wiring=wiring,
        backbone_units=[64, 32],
        backbone_layers=2,
        backbone_dropout=0.1
    )
    
    # Create input to initialize weights
    batch_size = 4
    seq_length = 10
    input_dim = 8
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    original_output = original_model(x)
    
    # Get config and weights
    config = original_model.get_config()
    weights = original_model.get_weights()
    
    # Create new model from config
    loaded_model = CfC.from_config(config)
    
    # Set weights
    loaded_model.build((batch_size, seq_length, input_dim))
    loaded_model.set_weights(weights)
    
    # Compare outputs
    loaded_output = loaded_model(x)
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)


def test_bidirectional():
    """Test bidirectional processing."""
    # Create model
    wiring = Random(units=16, output_dim=2, sparsity_level=0.4)
    wiring.build(8)
    base_model = LTC(wiring=wiring)
    model = keras.layers.Bidirectional(base_model, merge_mode="concat")
    
    # Create data
    batch_size = 4
    seq_length = 10
    input_dim = 8
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, seq_length, 4)  # Double output size due to concat


def test_time_handling():
    """Test time-aware processing."""
    # Create model
    wiring = Random(units=16, output_dim=1, sparsity_level=0.3)
    wiring.build(8)
    model = LTC(wiring=wiring)
    
    # Create data
    batch_size = 8
    seq_length = 15
    input_dim = 8
    x = np.random.normal(size=(batch_size, seq_length, input_dim)).astype(np.float32)
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, seq_length, 1)
