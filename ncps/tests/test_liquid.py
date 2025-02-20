"""Tests for liquid neurons."""

import pytest
import numpy as np
from ncps import ops

from ncps.layers.liquid import LiquidCell, CfCCell
from ncps.layers.rnn import RNN


def test_liquid_cell():
    """Test base liquid cell."""
    # Test without backbone
    cell = LiquidCell(32, activation="tanh")
    
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    
    states = cell.get_initial_state(batch_size)
    assert len(states) == 1
    assert states[0].shape == (batch_size, 32)
    
    y, new_states = cell(x, states)
    assert y.shape == (batch_size, input_dim + 32)  # concatenated features
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)
    
    # Test with backbone
    cell = LiquidCell(
        32,
        activation="tanh",
        backbone_units=64,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    
    states = cell.get_initial_state(batch_size)
    y, new_states = cell(x, states)
    assert y.shape == (batch_size, 64)  # backbone output size
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)
    
    # Test with training=True (dropout active)
    y1, _ = cell(x, states, training=True)
    y2, _ = cell(x, states, training=True)
    assert not np.allclose(y1, y2)
    
    # Test with training=False (no dropout)
    y1, _ = cell(x, states, training=False)
    y2, _ = cell(x, states, training=False)
    assert np.allclose(y1, y2)


def test_cfc_cell_pure():
    """Test CfC cell in pure mode."""
    # Create cell
    cell = CfCCell(32, mode="pure")
    
    # Test with batch input
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    t = 0.1
    
    # Get initial state
    states = cell.get_initial_state(batch_size)
    
    # Test step with scalar time
    y, new_states = cell([x, t], states)
    assert y.shape == (batch_size, input_dim + 32)  # concatenated features
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)
    
    # Test step with tensor time
    t = ops.ones((batch_size, 1)) * 0.1
    y, new_states = cell([x, t], states)
    assert y.shape == (batch_size, input_dim + 32)
    
    # Test solution stays bounded
    assert ops.reduce_max(ops.abs(new_states[0])) <= 2.0  # A is initialized to 1.0


def test_cfc_cell_gated():
    """Test CfC cell in gated mode."""
    # Create cell
    cell = CfCCell(32, mode="gated")
    
    # Test with batch input
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    t = 0.1
    
    # Get initial state
    states = cell.get_initial_state(batch_size)
    
    # Test step
    y, new_states = cell([x, t], states)
    assert y.shape == (batch_size, input_dim + 32)  # concatenated features
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)
    
    # Test gate values are between 0 and 1
    gate_output = ops.sigmoid(-t * (ops.matmul(states[0], cell.gate_kernel) + cell.gate_bias))
    assert np.all(gate_output >= 0)
    assert np.all(gate_output <= 1)


def test_cfc_cell_no_gate():
    """Test CfC cell in no-gate mode."""
    # Create cell
    cell = CfCCell(32, mode="no_gate")
    
    # Test with batch input
    batch_size = 8
    input_dim = 64
    x = ops.ones((batch_size, input_dim))
    t = 0.1
    
    # Get initial state
    states = cell.get_initial_state(batch_size)
    
    # Test step
    y, new_states = cell([x, t], states)
    assert y.shape == (batch_size, input_dim + 32)  # concatenated features
    assert len(new_states) == 1
    assert new_states[0].shape == (batch_size, 32)


def test_cfc_rnn():
    """Test CfC cell in RNN layer."""
    # Create layer
    cell = CfCCell(32, mode="pure")
    layer = RNN(cell, return_sequences=True)
    
    # Test with sequence input
    batch_size = 8
    seq_len = 10
    input_dim = 64
    x = ops.ones((batch_size, seq_len, input_dim))
    t = ops.ones((batch_size, seq_len, 1)) * 0.1
    
    # Test with return_sequences=True
    y = layer([x, t])
    assert y.shape == (batch_size, seq_len, input_dim + 32)  # concatenated features
    
    # Test with return_sequences=False
    layer = RNN(cell, return_sequences=False)
    y = layer([x, t])
    assert y.shape == (batch_size, input_dim + 32)
    
    # Test solution stays bounded
    assert ops.reduce_max(ops.abs(y[:, -32:])) <= 2.0  # Check state part


def test_serialization():
    """Test cell serialization."""
    # Test LiquidCell
    cell = LiquidCell(
        32,
        activation="tanh",
        backbone_units=64,
        backbone_layers=2
    )
    config = cell.get_config()
    new_cell = LiquidCell.from_config(config)
    assert new_cell.units == cell.units
    assert new_cell.backbone_units == cell.backbone_units
    
    # Test CfCCell
    cell = CfCCell(32, mode="pure")
    config = cell.get_config()
    new_cell = CfCCell.from_config(config)
    assert new_cell.units == cell.units
    assert new_cell.mode == cell.mode