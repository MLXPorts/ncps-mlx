"""Tests for CfC cell implementation."""

import pytest
import keras
import numpy as np
from keras import regularizers, constraints
from ncps.layers.cfc import CfCCell


def test_initialization():
    """Test CfC initialization."""
    # Test default configuration
    cell = CfCCell(32)
    assert cell.units == 32
    assert cell.mode == "default"
    assert cell.use_bias is True
    assert cell.dropout == 0.0
    
    # Test custom configuration
    cell = CfCCell(
        64,
        mode="pure",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.L1(0.01),
        dropout=0.5,
        seed=42
    )
    assert cell.units == 64
    assert cell.mode == "pure"
    assert cell.use_bias is False
    assert cell.dropout == 0.5
    assert cell.seed == 42
    
    # Test invalid mode
    with pytest.raises(ValueError):
        CfCCell(32, mode="invalid")
    
    # Test invalid units
    with pytest.raises(ValueError):
        CfCCell(-1)


def test_build():
    """Test build method."""
    # Test without bias
    cell = CfCCell(32, use_bias=False)
    cell.build((None, 64))
    assert cell.built
    assert cell.kernel.shape == (96, 32)  # input_dim + units, units
    assert not hasattr(cell, "bias")
    
    # Test with bias
    cell = CfCCell(32, use_bias=True)
    cell.build((None, 64))
    assert cell.built
    assert cell.kernel.shape == (96, 32)
    assert cell.bias.shape == (32,)
    
    # Test pure mode
    cell = CfCCell(32, mode="pure")
    cell.build((None, 64))
    assert hasattr(cell, "w_tau")
    assert hasattr(cell, "A")
    
    # Test gated modes
    for mode in ["default", "no_gate"]:
        cell = CfCCell(32, mode=mode)
        cell.build((None, 64))
        assert hasattr(cell, "gate_kernel")
        if cell.use_bias:
            assert hasattr(cell, "gate_bias")


def test_constraints():
    """Test weight constraints."""
    constraint = constraints.MaxNorm(2.0)
    cell = CfCCell(
        32,
        kernel_constraint=constraint,
        bias_constraint=constraint
    )
    cell.build((None, 64))
    
    assert cell.kernel.constraint == constraint
    if cell.use_bias:
        assert cell.bias.constraint == constraint


def test_regularization():
    """Test weight regularization."""
    regularizer = regularizers.L1(0.01)
    cell = CfCCell(
        32,
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer
    )
    cell.build((None, 64))
    
    assert cell.losses
    num_expected_losses = 2 if cell.use_bias else 1
    assert len(cell.losses) == num_expected_losses


def test_time_handling():
    """Test time input handling."""
    batch_size = 8
    input_dim = 64
    
    # Create inputs with different time scales
    x = keras.ops.ones((batch_size, input_dim))
    t1 = keras.ops.ones((batch_size, 1))  # Normal time step
    t2 = t1 * 0.1  # Much smaller time step
    state = [keras.ops.zeros((batch_size, 32))]
    
    # Create cell
    cell = CfCCell(32)
    cell.build((None, input_dim))
    
    # Test with different time steps
    output1, _ = cell([x, t1], state)
    output2, _ = cell([x, t2], state)
    
    # Outputs should differ due to time scaling
    assert output1.shape == (batch_size, 32)
    assert output2.shape == (batch_size, 32)
    assert not keras.ops.all(keras.ops.isclose(output1, output2))
    
    # Test without time (should default to 1.0)
    output3, _ = cell(x, state)
    assert keras.ops.all(keras.ops.isclose(output3, output1))


def test_dropout():
    """Test dropout behavior."""
    batch_size = 8
    input_dim = 64
    dropout_rate = 0.5
    
    # Create cell with high dropout
    cell = CfCCell(32, dropout=dropout_rate)
    cell.build((None, input_dim))
    
    # Create inputs
    x = keras.ops.ones((batch_size, input_dim))
    state = [keras.ops.zeros((batch_size, 32))]
    
    # Test training phase (with dropout)
    output1, _ = cell(x, state, training=True)
    output2, _ = cell(x, state, training=True)
    
    # Outputs should differ due to dropout
    assert not keras.ops.all(keras.ops.isclose(output1, output2))
    
    # Test inference phase (no dropout)
    output3, _ = cell(x, state, training=False)
    output4, _ = cell(x, state, training=False)
    
    # Outputs should be the same without dropout
    assert keras.ops.all(keras.ops.isclose(output3, output4))


def test_modes():
    """Test all CfC modes."""
    batch_size = 8
    input_dim = 64
    time_steps = 10
    inputs = keras.ops.ones((batch_size, time_steps, input_dim))
    
    # Test each mode
    modes = ["default", "pure", "no_gate"]
    outputs = []
    
    for mode in modes:
        cell = CfCCell(32, mode=mode)
        rnn = keras.layers.RNN(cell)
        output = rnn(inputs)
        outputs.append(output)
        
        # Check shapes
        assert output.shape == (batch_size, 32)
    
    # Different modes should give different results
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            assert not keras.ops.all(keras.ops.isclose(outputs[i], outputs[j]))


def test_training():
    """Test training functionality."""
    # Create model
    model = keras.Sequential([
        keras.layers.RNN(CfCCell(32)),
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


def test_stateful():
    """Test stateful operation."""
    # Create stateful RNN
    cell = CfCCell(32)
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
    assert not keras.ops.all(keras.ops.isclose(out1, out2))


def test_serialization():
    """Test serialization."""
    # Create and build cell
    cell = CfCCell(
        32,
        mode="pure",
        activation="relu",
        use_bias=False,
        kernel_regularizer=regularizers.L1(0.01),
        dropout=0.5,
        seed=42
    )
    cell.build((None, 64))
    
    # Get config
    config = keras.saving.serialize_keras_object(cell)
    
    # Reconstruct
    new_cell = keras.saving.deserialize_keras_object(config)
    
    # Should have same configuration
    assert new_cell.units == cell.units
    assert new_cell.mode == cell.mode
    assert new_cell.use_bias == cell.use_bias
    assert new_cell.dropout == cell.dropout
    assert new_cell.seed == cell.seed