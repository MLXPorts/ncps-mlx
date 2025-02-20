"""Tests for LTC cell implementation."""

import pytest
import keras
import numpy as np
from ncps.layers.ltc import LTCCell


def test_initialization():
    """Test LTC initialization."""
    # Test default configuration
    cell = LTCCell(32)
    assert cell.units == 32
    assert cell.backbone_units is None
    assert cell.backbone_layers == 0
    assert cell.backbone_dropout == 0.0
    
    # Test custom configuration
    cell = LTCCell(
        64,
        activation="relu",
        backbone_units=128,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    assert cell.units == 64
    assert cell.backbone_units == 128
    assert cell.backbone_layers == 2
    assert cell.backbone_dropout == 0.1


def test_build():
    """Test build method."""
    # Test without backbone
    cell = LTCCell(32)
    cell.build((None, 64))
    assert cell.built
    assert cell.kernel.shape == (96, 32)  # input_dim + units
    
    # Test with backbone
    cell = LTCCell(32, backbone_units=128, backbone_layers=1)
    cell.build((None, 64))
    assert cell.built
    assert cell.kernel.shape == (128, 32)  # backbone_units


def test_time_constants():
    """Test time constant behavior."""
    batch_size = 8
    input_dim = 64
    
    # Create inputs
    x = keras.ops.ones((batch_size, input_dim))
    t = keras.ops.ones((batch_size, 1)) * 0.5  # Half timestep
    state = [keras.ops.zeros((batch_size, 32))]
    
    # Create cell
    cell = LTCCell(32)
    cell.build((None, input_dim))
    
    # Test with different time steps
    output1, state1 = cell([x, t], state)
    output2, state2 = cell([x, t * 2], state)  # Double the time step
    
    # Outputs should differ due to time scaling
    assert output1.shape == (batch_size, 32)
    assert output2.shape == (batch_size, 32)
    assert not keras.ops.allclose(output1, output2)


def test_backbone():
    """Test backbone functionality."""
    batch_size = 8
    input_dim = 64
    time_steps = 10
    inputs = keras.ops.ones((batch_size, time_steps, input_dim))
    
    # Create cell with backbone
    cell = LTCCell(
        32,
        backbone_units=128,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    rnn = keras.layers.RNN(cell)
    
    # Test training (with dropout)
    output1 = rnn(inputs, training=True)
    assert output1.shape == (batch_size, 32)
    
    # Test inference (without dropout)
    output2 = rnn(inputs, training=False)
    assert output2.shape == (batch_size, 32)
    
    # Outputs should differ due to dropout
    assert not keras.ops.allclose(output1, output2)


def test_training():
    """Test training functionality."""
    # Create model
    model = keras.Sequential([
        keras.layers.RNN(LTCCell(32)),
        keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    
    # Create sample data
    batch_size = 8
    time_steps = 10
    features = 64
    x = np.random.normal(size=(batch_size, time_steps, features))
    y = np.random.normal(size=(batch_size, 1))
    
    # Should train without errors
    history = model.fit(x, y, epochs=1)
    assert len(history.history["loss"]) == 1


def test_serialization():
    """Test serialization."""
    # Create and build cell
    cell = LTCCell(
        32,
        backbone_units=128,
        backbone_layers=2
    )
    cell.build((None, 64))
    
    # Get config
    config = keras.saving.serialize_keras_object(cell)
    
    # Reconstruct
    new_cell = keras.saving.deserialize_keras_object(config)
    
    # Should have same configuration
    assert new_cell.units == cell.units
    assert new_cell.backbone_units == cell.backbone_units
    assert new_cell.backbone_layers == cell.backbone_layers


def test_stateful():
    """Test stateful operation."""
    # Create stateful RNN
    cell = LTCCell(32)
    rnn = keras.layers.RNN(cell, stateful=True)
    
    # Create inputs
    batch_size = 8
    time_steps = 10
    features = 64
    x = keras.ops.ones((batch_size, time_steps, features))
    
    # First call
    out1 = rnn(x)
    
    # Second call should use final state from first call
    out2 = rnn(x)
    
    # Outputs should differ due to state
    assert not keras.ops.allclose(out1, out2)


def test_activation():
    """Test activation function."""
    batch_size = 8
    input_dim = 64
    time_steps = 10
    inputs = keras.ops.ones((batch_size, time_steps, input_dim))
    
    # Test different activations
    activations = ["tanh", "relu", "sigmoid"]
    
    for act in activations:
        cell = LTCCell(32, activation=act)
        rnn = keras.layers.RNN(cell)
        output = rnn(inputs)
        assert output.shape == (batch_size, 32)