"""Tests for MLX liquid neuron implementations."""

import mlx.core as mx
import mlx.nn as nn
from ncps.mlx import CfC, LTCCell, LTC, CfCCell
from ncps import wirings
import numpy as np
import unittest
import tempfile
import os

from ncps.mlx import save_model, load_model


def generate_data(N):
    """Generate synthetic time-series data."""
    data_x = mx.array(
        np.stack(
            [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
            axis=1,
        ),
        dtype=mx.float32,
    )[None, :, :]  # Add batch dimension
    data_y = mx.array(
        np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]), dtype=mx.float32
    )
    return data_x, data_y


class TestLiquidNeurons(unittest.TestCase):
    """Test suite for liquid neuron implementations."""
    
    def test_base_cell_functionality(self):
        """Test basic functionality of liquid cells."""
        batch_size = 32
        input_dim = 10
        hidden_size = 20
        
        # Test CfCCell
        cfc_cell = CfCCell(input_size=input_dim, hidden_size=hidden_size)
        x = mx.random.normal((batch_size, input_dim))
        state = mx.zeros((batch_size, hidden_size))
        output, new_state = cfc_cell(x, state)
        self.assertEqual(output.shape, (batch_size, hidden_size))
        self.assertEqual(new_state.shape, (batch_size, hidden_size))
        
        # Test LTCCell
        ltc_cell = LTCCell(input_size=input_dim, hidden_size=hidden_size)
        output, new_state = ltc_cell(x, state)
        self.assertEqual(output.shape, (batch_size, hidden_size))
        self.assertEqual(new_state.shape, (batch_size, hidden_size))
        
    def test_time_aware_processing(self):
        """Test time-aware processing in liquid cells."""
        batch_size = 32
        seq_len = 15
        input_dim = 10
        hidden_size = 20
        
        # Test CfC with time delta
        cfc = CfC(
            input_size=input_dim,
            hidden_size=hidden_size,
            return_sequences=True
        )
        x = mx.random.normal((batch_size, seq_len, input_dim))
        time_delta = mx.ones((batch_size, seq_len, 1))
        output = cfc(x, time_delta=time_delta)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test LTC with time delta
        ltc = LTC(
            input_size=input_dim,
            hidden_size=hidden_size,
            return_sequences=True
        )
        output = ltc(x, time_delta=time_delta)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
    def test_bidirectional_processing(self):
        """Test bidirectional processing in liquid RNNs."""
        batch_size = 32
        seq_len = 15
        input_dim = 10
        hidden_size = 20
        
        # Test bidirectional CfC
        cfc = CfC(
            input_size=input_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            return_sequences=True
        )
        x = mx.random.normal((batch_size, seq_len, input_dim))
        output = cfc(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size * 2))
        
        # Test bidirectional LTC
        ltc = LTC(
            input_size=input_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            return_sequences=True
        )
        output = ltc(x)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size * 2))
        
    def test_backbone_layers(self):
        """Test backbone layers in liquid cells."""
        batch_size = 32
        input_dim = 10
        hidden_size = 20
        backbone_units = 64
        backbone_layers = 2
        
        # Test CfC with backbone
        cfc_cell = CfCCell(
            input_size=input_dim,
            hidden_size=hidden_size,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers
        )
        x = mx.random.normal((batch_size, input_dim))
        state = mx.zeros((batch_size, hidden_size))
        output, new_state = cfc_cell(x, state)
        self.assertEqual(output.shape, (batch_size, hidden_size))
        
        # Test LTC with backbone
        ltc_cell = LTCCell(
            input_size=input_dim,
            hidden_size=hidden_size,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers
        )
        output, new_state = ltc_cell(x, state)
        self.assertEqual(output.shape, (batch_size, hidden_size))
        
    def test_state_handling(self):
        """Test state handling in liquid RNNs."""
        batch_size = 32
        seq_len = 15
        input_dim = 10
        hidden_size = 20
        num_layers = 2
        
        # Test CfC state handling
        cfc = CfC(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            return_state=True
        )
        x = mx.random.normal((batch_size, seq_len, input_dim))
        output, states = cfc(x)
        self.assertEqual(len(states), num_layers)
        for state in states:
            self.assertEqual(state.shape, (batch_size, hidden_size))
            
        # Test LTC state handling
        ltc = LTC(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            return_state=True
        )
        output, states = ltc(x)
        self.assertEqual(len(states), num_layers)
        for state in states:
            self.assertEqual(state.shape, (batch_size, hidden_size))
            
    def test_model_serialization(self):
        """Test model saving and loading."""
        batch_size = 32
        seq_len = 15
        input_dim = 10
        hidden_size = 20
        
        # Test CfC serialization
        cfc = CfC(
            input_size=input_dim,
            hidden_size=hidden_size,
            return_sequences=True
        )
        x = mx.random.normal((batch_size, seq_len, input_dim))
        original_output = cfc(x)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            save_model(cfc, tmp.name)
            loaded_cfc = CfC(
                input_size=input_dim,
                hidden_size=hidden_size,
                return_sequences=True
            )
            load_model(loaded_cfc, tmp.name)
        os.unlink(tmp.name)
        
        loaded_output = loaded_cfc(x)
        self.assertTrue(mx.allclose(original_output, loaded_output))
        
        # Test LTC serialization
        ltc = LTC(
            input_size=input_dim,
            hidden_size=hidden_size,
            return_sequences=True
        )
        original_output = ltc(x)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            save_model(ltc, tmp.name)
            loaded_ltc = LTC(
                input_size=input_dim,
                hidden_size=hidden_size,
                return_sequences=True
            )
            load_model(loaded_ltc, tmp.name)
        os.unlink(tmp.name)
        
        loaded_output = loaded_ltc(x)
        self.assertTrue(mx.allclose(original_output, loaded_output))


if __name__ == '__main__':
    unittest.main()
