"""Tests for base cell implementation."""

import pytest
import keras
import numpy as np
from ncps import ops

from ncps.layers.base import BackboneLayerCell


class DummyWiring:
    """Dummy wiring for testing."""
    
    def __init__(self, units, output_dim=None):
        self.units = units
        self.output_dim = output_dim or units
    
    def get_config(self):
        return {
            "units": self.units,
            "output_dim": self.output_dim
        }


class TestCell(BackboneLayerCell):
    """Test cell implementation."""
    
    def call(
        self,
        inputs,
        states,
        training=None,
        **kwargs
    ):
        """Process one timestep."""
        # Handle time input
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
            t = inputs[1]
            # Convert scalar time to tensor
            if isinstance(t, (int, float)):
                t = ops.ones_like(x[:, :1]) * t
            t = ops.reshape(t, [-1, 1])
        else:
            x = inputs
            t = ops.ones_like(x[:, :1])  # Default time delta
            
        # Get current state
        h = states[0]
        
        # Combine inputs and state
        concat = ops.concatenate([x, h], axis=-1)
        
        # Apply backbone if present
        if self.backbone is not None:
            features = self.apply_backbone(
                concat,
                self.backbone,
                training=training
            )
        else:
            features = concat
            
        # Simple linear transformation
        output = self.activation(features)
        
        # Return same size state as input state
        new_state = output[:, :self.state_size]
        
        return output, [new_state]
    
    @staticmethod
    def _get_wiring_class():
        return DummyWiring


def test_initialization():
    """Test cell initialization."""
    # Test default configuration
    wiring = DummyWiring(32)
    cell = TestCell(wiring)
    
    assert cell.units == 32
    assert cell.state_size == 32
    assert cell.output_size == 32
    assert not cell.built
    
    # Test with backbone
    cell = TestCell(
        wiring,
        activation="relu",
        backbone_units=64,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    
    assert cell.backbone_units == [64, 64]
    assert cell.backbone_layers == 2
    assert cell.backbone_dropout == 0.1
    assert cell.activation_name == "relu"


def test_build():
    """Test build method."""
    wiring = DummyWiring(32)
    
    # Test without backbone
    cell = TestCell(wiring)
    cell.build((None, 64))
    
    assert cell.built
    assert cell.input_size == 64
    assert cell.backbone is None
    assert cell.backbone_output_dim == 96  # input_dim + state_size
    
    # Test with backbone
    cell = TestCell(
        wiring,
        backbone_units=128,
        backbone_layers=1
    )
    cell.build((None, 64))
    
    assert cell.built
    assert len(cell.backbone) == 1
    assert cell.backbone_output_dim == 128


def test_get_initial_state():
    """Test initial state generation."""
    wiring = DummyWiring(32)
    cell = TestCell(wiring)
    
    # Test with batch size
    states = cell.get_initial_state(batch_size=8)
    assert len(states) == 1
    assert states[0].shape == (8, 32)
    
    # Test with dtype
    states = cell.get_initial_state(batch_size=8, dtype="float64")
    assert states[0].dtype == "float64"


def test_time_handling():
    """Test time input processing."""
    wiring = DummyWiring(32)
    cell = TestCell(wiring)
    cell.build((None, 64))
    
    batch_size = 8
    input_dim = 64
    
    # Create inputs
    x = ops.ones((batch_size, input_dim))
    state = [ops.zeros((batch_size, 32))]
    
    # Test without time
    output1, new_state1 = cell(x, state, training=False)
    assert output1.shape == (batch_size, 96)  # concat size
    assert len(new_state1) == 1
    assert new_state1[0].shape == (batch_size, 32)
    
    # Test with scalar time
    t = ops.convert_to_tensor(0.5)
    output2, new_state2 = cell([x, t], state, training=False)
    assert output2.shape == (batch_size, 96)
    assert new_state2[0].shape == (batch_size, 32)
    
    # Test with tensor time
    t = ops.ones((batch_size, 1))
    output3, new_state3 = cell([x, t], state, training=False)
    assert output3.shape == (batch_size, 96)
    assert new_state3[0].shape == (batch_size, 32)


def test_backbone():
    """Test backbone network."""
    wiring = DummyWiring(32)
    cell = TestCell(
        wiring,
        backbone_units=128,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    cell.build((None, 64))
    
    batch_size = 8
    input_dim = 64
    
    # Create inputs
    x = ops.ones((batch_size, input_dim))
    state = [ops.zeros((batch_size, 32))]
    
    # Test training phase
    output1, _ = cell(x, state, training=True)
    output2, _ = cell(x, state, training=True)
    
    # Outputs should differ due to dropout
    assert not np.allclose(output1.numpy(), output2.numpy())
    
    # Test inference phase
    output3, _ = cell(x, state, training=False)
    output4, _ = cell(x, state, training=False)
    
    # Outputs should be the same without dropout
    assert np.allclose(output3.numpy(), output4.numpy())


def test_serialization():
    """Test cell serialization."""
    wiring = DummyWiring(32, output_dim=16)
    cell = TestCell(
        wiring,
        activation="relu",
        backbone_units=64,
        backbone_layers=2
    )
    config = cell.get_config()
    
    # Check config
    assert config["wiring"]["units"] == 32
    assert config["wiring"]["output_dim"] == 16
    assert config["activation"] == "relu"
    assert config["backbone_units"] == [64, 64]
    assert config["backbone_layers"] == 2
    
    # Reconstruct from config
    new_cell = TestCell.from_config(config)
    
    # Check reconstruction
    assert new_cell.units == cell.units
    assert new_cell.output_size == cell.output_size
    assert new_cell.activation_name == cell.activation_name
    assert new_cell.backbone_units == cell.backbone_units
    assert new_cell.backbone_layers == cell.backbone_layers