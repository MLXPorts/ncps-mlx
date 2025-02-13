"""Tests for PyTorch neural circuit implementations."""

import torch
import unittest
import numpy as np
from ncps.torch import LiquidCell, LiquidRNN
from ncps import wirings


def generate_data(batch_size=32, seq_len=15, input_dim=10):
    """Generate synthetic time-series data."""
    return torch.randn(batch_size, seq_len, input_dim)


class TestLiquidCell(unittest.TestCase):
    """Test suite for liquid cell implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 10
        self.hidden_size = 20
        self.wiring = wirings.Random(
            self.hidden_size,
            output_dim=self.hidden_size,
            sparsity_level=0.5
        )
        
    def test_base_cell_functionality(self):
        """Test basic functionality of liquid cells."""
        # Test with default parameters
        cell = LiquidCell(self.wiring)
        x = torch.randn(self.batch_size, self.input_dim)
        state = torch.zeros(self.batch_size, self.hidden_size)
        output, new_state = cell(x, state)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(new_state.shape, (self.batch_size, self.hidden_size))
        
        # Test with backbone network
        cell_with_backbone = LiquidCell(
            self.wiring,
            backbone_units=[32, 64],
            backbone_layers=2,
            backbone_dropout=0.1
        )
        output, new_state = cell_with_backbone(x, state)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(new_state.shape, (self.batch_size, self.hidden_size))
        
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["tanh", "relu", "sigmoid"]
        for activation in activations:
            cell = LiquidCell(self.wiring, activation=activation)
            x = torch.randn(self.batch_size, self.input_dim)
            state = torch.zeros(self.batch_size, self.hidden_size)
            output, new_state = cell(x, state)
            
            self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
            
    def test_backbone_configurations(self):
        """Test various backbone network configurations."""
        configs = [
            {"units": [32], "layers": 1},
            {"units": [32, 64], "layers": 2},
            {"units": 32, "layers": 3},  # Should create [32, 32, 32]
        ]
        
        for config in configs:
            cell = LiquidCell(
                self.wiring,
                backbone_units=config["units"],
                backbone_layers=config["layers"]
            )
            x = torch.randn(self.batch_size, self.input_dim)
            state = torch.zeros(self.batch_size, self.hidden_size)
            output, new_state = cell(x, state)
            
            self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
            
    def test_serialization(self):
        """Test state dict saving and loading."""
        # Create and initialize original cell
        original_cell = LiquidCell(
            self.wiring,
            backbone_units=[32, 64],
            backbone_layers=2
        )
        x = torch.randn(self.batch_size, self.input_dim)
        state = torch.zeros(self.batch_size, self.hidden_size)
        original_output, _ = original_cell(x, state)
        
        # Save state dict
        state_dict = original_cell.state_dict()
        
        # Create new cell and load state
        loaded_cell = LiquidCell(
            self.wiring,
            backbone_units=[32, 64],
            backbone_layers=2
        )
        loaded_cell.load_state_dict(state_dict)
        
        # Compare outputs
        loaded_output, _ = loaded_cell(x, state)
        self.assertTrue(torch.allclose(original_output, loaded_output))


class TestLiquidRNN(unittest.TestCase):
    """Test suite for liquid RNN implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.seq_len = 15
        self.input_dim = 10
        self.hidden_size = 20
        self.wiring = wirings.Random(
            self.hidden_size,
            output_dim=self.hidden_size,
            sparsity_level=0.5
        )
        self.cell = LiquidCell(self.wiring)
        
    def test_sequence_processing(self):
        """Test basic sequence processing."""
        rnn = LiquidRNN(self.cell)
        x = generate_data(
            self.batch_size,
            self.seq_len,
            self.input_dim
        )
        output = rnn(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        
        # Test with return sequences
        rnn = LiquidRNN(self.cell, return_sequences=True)
        output = rnn(x)
        
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.hidden_size)
        )
        
    def test_state_handling(self):
        """Test state handling."""
        rnn = LiquidRNN(self.cell, return_state=True)
        x = generate_data(
            self.batch_size,
            self.seq_len,
            self.input_dim
        )
        output, final_state = rnn(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(
            final_state.shape,
            (self.batch_size, self.hidden_size)
        )
        
        # Test with initial state
        initial_state = torch.zeros(self.batch_size, self.hidden_size)
        output, final_state = rnn(x, initial_states=[initial_state])
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(
            final_state.shape,
            (self.batch_size, self.hidden_size)
        )
        
    def test_bidirectional(self):
        """Test bidirectional processing."""
        rnn = LiquidRNN(
            self.cell,
            bidirectional=True,
            merge_mode="concat"
        )
        x = generate_data(
            self.batch_size,
            self.seq_len,
            self.input_dim
        )
        output = rnn(x)
        
        self.assertEqual(
            output.shape,
            (self.batch_size, self.hidden_size * 2)
        )
        
        # Test different merge modes
        merge_modes = ["sum", "mul", "ave"]
        for mode in merge_modes:
            rnn = LiquidRNN(
                self.cell,
                bidirectional=True,
                merge_mode=mode
            )
            output = rnn(x)
            
            self.assertEqual(
                output.shape,
                (self.batch_size, self.hidden_size)
            )
            
    def test_time_delta(self):
        """Test time-aware processing."""
        rnn = LiquidRNN(self.cell)
        x = generate_data(
            self.batch_size,
            self.seq_len,
            self.input_dim
        )
        
        # Test with scalar time delta
        output = rnn(x, time_delta=0.1)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        
        # Test with tensor time delta
        time_delta = torch.ones(self.batch_size, self.seq_len) * 0.1
        output = rnn(x, time_delta=time_delta)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))


if __name__ == "__main__":
    unittest.main()