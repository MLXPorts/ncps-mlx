"""Tests for basic layer system."""

import pytest
import keras
import numpy as np
from keras import ops

from ncps.layers.layer import Layer, Dense, Sequential
from ncps.layers.rnn import RNNCell, RNN, register_cell


@register_cell
class SimpleRNNCell(RNNCell):
    """Simple RNN cell for testing."""
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="kernel"
        )
        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer="glorot_uniform",
            name="recurrent_kernel"
        )
        
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="bias"
        )
        
        self.built = True
    
    def call(self, inputs, states, training=None):
        prev_h = states[0]
        h = ops.matmul(inputs, self.kernel)
        h = h + ops.matmul(prev_h, self.recurrent_kernel)
        h = h + self.bias
        h = ops.tanh(h)
        return h, [h]
    
    def get_config(self):
        config = super().get_config()
        return config


def test_dense_layer():
    """Test Dense layer."""
    # Create layer
    layer = Dense(32, activation="relu")
    
    # Test with batch input
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    
    # First call builds the layer
    y = layer(x)
    
    # Check output
    assert y.shape == (batch_size, 32)
    assert layer.built
    assert len(layer.trainable_weights) == 2  # kernel and bias
    
    # Test without bias
    layer = Dense(32, use_bias=False)
    y = layer(x)
    assert len(layer.trainable_weights) == 1  # just kernel


def test_sequential():
    """Test Sequential container."""
    # Create model
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16)
    ])
    
    # Test forward pass
    batch_size = 8
    input_dim = 128
    x = ops.ones((batch_size, input_dim))
    y = model(x)
    
    # Check output
    assert y.shape == (batch_size, 16)
    
    # Check weights
    weights = model.get_weights()
    assert len(weights) == 6  # 2 weights per layer (kernel + bias)
    
    # Test weight setting
    model.set_weights(weights)


def test_rnn_cell():
    """Test RNN cell."""
    # Create cell
    cell = SimpleRNNCell(32)
    
    # Test with batch input
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    
    # Get initial state
    states = cell.get_initial_state(batch_size)
    assert len(states) == 1
    assert states[0].shape == (batch_size, 32)
    
    # Test step
    y, new_states = cell(x, states)
    assert y.shape == (batch_size, 32)
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)


def test_rnn_layer():
    """Test RNN layer."""
    # Create layer
    cell = SimpleRNNCell(32)
    layer = RNN(cell, return_sequences=True)
    
    # Test with sequence input
    batch_size = 8
    seq_len = 10
    input_dim = 64
    x = ops.ones((batch_size, seq_len, input_dim))
    
    # Test with return_sequences=True
    y = layer(x)
    assert y.shape == (batch_size, seq_len, 32)
    
    # Test with return_sequences=False
    layer = RNN(cell, return_sequences=False)
    y = layer(x)
    assert y.shape == (batch_size, 32)
    
    # Test with return_state=True
    layer = RNN(cell, return_sequences=False, return_state=True)
    y, states = layer(x)
    assert y.shape == (batch_size, 32)
    assert len(states) == 1
    assert states[0].shape == (batch_size, 32)
    
    # Test with initial state
    initial_state = [ops.zeros((batch_size, 32))]
    y = layer(x, initial_state=initial_state)


def test_serialization():
    """Test layer serialization."""
    # Test Dense
    dense = Dense(32, activation="relu")
    config = dense.get_config()
    new_dense = Dense.from_config(config)
    assert new_dense.units == dense.units
    
    # Test RNN
    cell = SimpleRNNCell(32)
    rnn = RNN(cell, return_sequences=True)
    
    # Build layer to create weights
    batch_size = 8
    seq_len = 10
    input_dim = 64
    x = ops.ones((batch_size, seq_len, input_dim))
    rnn(x)  # Forward pass builds the layer
    
    config = rnn.get_config()
    new_rnn = RNN.from_config(config)
    assert new_rnn.return_sequences == rnn.return_sequences
    assert new_rnn.cell.units == rnn.cell.units