"""Tests for liquid neural network utilities."""

import pytest
import keras
import numpy as np
from keras import ops, layers

from ncps.layers.liquid_utils import (
    TimeAwareMixin,
    BackboneMixin,
    get_activation,
    lecun_tanh,
    ensure_time_dim,
    broadcast_to_batch
)


class TestTimeAwareMixin:
    """Test time handling mixin."""
    
    class DummyCell(TimeAwareMixin):
        """Dummy cell for testing."""
        pass
    
    def test_process_time_delta_none(self):
        """Test processing None time delta."""
        cell = self.DummyCell()
        batch_size = 8
        seq_len = 10
        
        # Test without sequence
        time = cell.process_time_delta(None, batch_size)
        assert time.shape == (batch_size, 1)
        assert np.all(time.numpy() == 1.0)
        
        # Test with sequence
        time = cell.process_time_delta(None, batch_size, seq_len)
        assert time.shape == (batch_size, seq_len, 1)
        assert np.all(time.numpy() == 1.0)
    
    def test_process_time_delta_scalar(self):
        """Test processing scalar time delta."""
        cell = self.DummyCell()
        batch_size = 8
        seq_len = 10
        dt = 0.1
        
        # Test without sequence
        time = cell.process_time_delta(dt, batch_size)
        assert time.shape == (batch_size, 1)
        assert np.all(time.numpy() == dt)
        
        # Test with sequence
        time = cell.process_time_delta(dt, batch_size, seq_len)
        assert time.shape == (batch_size, seq_len, 1)
        assert np.all(time.numpy() == dt)
    
    def test_process_time_delta_tensor(self):
        """Test processing tensor time delta."""
        cell = self.DummyCell()
        batch_size = 8
        seq_len = 10
        
        # Test 1D time
        dt = ops.ones(batch_size)
        time = cell.process_time_delta(dt, batch_size)
        assert time.shape == (batch_size, 1)
        
        # Test 2D time
        dt = ops.ones((batch_size, seq_len))
        time = cell.process_time_delta(dt, batch_size, seq_len)
        assert time.shape == (batch_size, seq_len, 1)


class TestBackboneMixin:
    """Test backbone network mixin."""
    
    class DummyCell(BackboneMixin):
        """Dummy cell for testing."""
        pass
    
    def test_build_backbone(self):
        """Test building backbone layers."""
        cell = self.DummyCell()
        input_size = 64
        backbone_units = 128
        backbone_layers = 2
        backbone_dropout = 0.1
        
        # Build backbone
        backbone = cell.build_backbone(
            input_size,
            backbone_units,
            backbone_layers,
            backbone_dropout,
            "tanh"
        )
        
        # Check number of layers
        assert len(backbone) == 4  # 2 dense + 2 dropout
        
        # Check layer types
        assert isinstance(backbone[0], layers.Dense)
        assert isinstance(backbone[1], layers.Dropout)
        assert isinstance(backbone[2], layers.Dense)
        assert isinstance(backbone[3], layers.Dropout)
        
        # Check layer configs
        assert backbone[0].units == backbone_units
        assert backbone[1].rate == backbone_dropout
    
    def test_apply_backbone(self):
        """Test applying backbone to input."""
        cell = self.DummyCell()
        batch_size = 8
        input_size = 64
        backbone_units = 128
        
        # Create input
        x = ops.ones((batch_size, input_size))
        
        # Build and apply backbone
        backbone = cell.build_backbone(
            input_size,
            backbone_units,
            1,
            0.0,
            "tanh"
        )
        output = cell.apply_backbone(x, backbone)
        
        # Check output shape
        assert output.shape == (batch_size, backbone_units)


def test_lecun_tanh():
    """Test LeCun tanh activation."""
    x = ops.ones((8, 32))
    y = lecun_tanh(x)
    
    # Check shape
    assert y.shape == x.shape
    
    # Check values
    expected = 1.7159 * np.tanh(0.666)
    assert np.allclose(y.numpy(), expected)


def test_get_activation():
    """Test activation function getter."""
    # Test valid activations
    valid = ["lecun_tanh", "tanh", "relu", "gelu", "sigmoid", "linear"]
    for name in valid:
        fn = get_activation(name)
        assert callable(fn)
        
        # Test with dummy input
        x = ops.ones((8, 32))
        y = fn(x)
        assert y.shape == x.shape
    
    # Test invalid activation
    with pytest.raises(ValueError):
        get_activation("invalid")


def test_ensure_time_dim():
    """Test time dimension handling."""
    # Test 2D input
    x = ops.ones((8, 32))
    y = ensure_time_dim(x)
    assert y.shape == (8, 1, 32)
    
    # Test 3D input
    x = ops.ones((8, 10, 32))
    y = ensure_time_dim(x)
    assert y.shape == (8, 10, 32)


def test_broadcast_to_batch():
    """Test batch broadcasting."""
    batch_size = 8
    
    # Test 1D input
    x = ops.ones(32)
    y = broadcast_to_batch(x, batch_size)
    assert y.shape == (batch_size, 32)
    
    # Test 2D input
    x = ops.ones((1, 32))
    y = broadcast_to_batch(x, batch_size)
    assert y.shape == (batch_size, 32)