"""Tests for advanced wiring patterns."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring


class SignalWiring(Wiring):
    """Signal processing wiring pattern."""
    
    def __init__(
        self,
        input_size: int,
        num_bands: int = 4,
        neurons_per_band: int = 16,
        output_size: int = 1
    ):
        total_units = num_bands * neurons_per_band + output_size
        super().__init__(total_units)
        
        # Store configuration
        self.num_bands = num_bands
        self.neurons_per_band = neurons_per_band
        self.output_size = output_size
        
        # Set output dimension
        self.set_output_dim(output_size)
        
        # Define band ranges
        self.band_ranges = [
            range(
                output_size + i * neurons_per_band,
                output_size + (i + 1) * neurons_per_band
            )
            for i in range(num_bands)
        ]
        
        # Build connectivity
        self._build_band_connections()
        self._build_cross_band_connections()
        self._build_output_connections()
    
    def _build_band_connections(self):
        """Build connections within each frequency band."""
        for band_range in self.band_ranges:
            for src in band_range:
                for dest in band_range:
                    if src != dest:  # No self-connections
                        self.add_synapse(src, dest, 1)
    
    def _build_cross_band_connections(self):
        """Build connections between adjacent frequency bands."""
        for i in range(self.num_bands - 1):
            current_band = self.band_ranges[i]
            next_band = self.band_ranges[i + 1]
            
            for src in current_band:
                for dest in np.random.choice(list(next_band), size=2, replace=False):
                    self.add_synapse(src, dest, 1)
    
    def _build_output_connections(self):
        """Build connections to output neurons."""
        output_range = range(self.output_size)
        
        for band_range in self.band_ranges:
            for src in band_range:
                for dest in output_range:
                    self.add_synapse(src, dest, 1)


class RoboticsWiring(Wiring):
    """Robotics wiring pattern."""
    
    def __init__(
        self,
        sensor_neurons: int,
        state_neurons: int,
        control_neurons: int,
        sensor_fanout: int = 4,
        state_recurrent: int = 3,
        control_fanin: int = 4
    ):
        total_units = sensor_neurons + state_neurons + control_neurons
        super().__init__(total_units)
        
        # Store configuration
        self.sensor_neurons = sensor_neurons
        self.state_neurons = state_neurons
        self.control_neurons = control_neurons
        self.sensor_fanout = sensor_fanout
        self.state_recurrent = state_recurrent
        self.control_fanin = control_fanin
        
        # Set output dimension
        self.set_output_dim(control_neurons)
        
        # Define ranges
        self.control_range = range(control_neurons)
        self.state_range = range(
            control_neurons,
            control_neurons + state_neurons
        )
        self.sensor_range = range(
            control_neurons + state_neurons,
            total_units
        )
        
        # Build connectivity
        self._build_sensor_connections()
        self._build_state_connections()
        self._build_control_connections()
    
    def _build_sensor_connections(self):
        """Build connections from sensor layer."""
        for src in self.sensor_range:
            targets = np.random.choice(
                list(self.state_range),
                size=self.sensor_fanout,
                replace=False
            )
            for dest in targets:
                self.add_synapse(src, dest, 1)
            
            if np.random.random() < 0.2:  # 20% chance of reflex connection
                dest = np.random.choice(list(self.control_range))
                self.add_synapse(src, dest, 1)
    
    def _build_state_connections(self):
        """Build connections in state estimation layer."""
        for _ in range(self.state_recurrent):
            src = np.random.choice(list(self.state_range))
            dest = np.random.choice(list(self.state_range))
            self.add_synapse(src, dest, 1)
    
    def _build_control_connections(self):
        """Build connections to control layer."""
        for dest in self.control_range:
            sources = np.random.choice(
                list(self.state_range),
                size=self.control_fanin,
                replace=False
            )
            for src in sources:
                self.add_synapse(src, dest, 1)


class AttentionWiring(Wiring):
    """Attention-based wiring pattern."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_size: int
    ):
        # Size calculations
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        total_units = hidden_size * 4  # For Q,K,V and output
        
        super().__init__(total_units)
        
        # Define ranges
        self.query_range = range(0, hidden_size)
        self.key_range = range(hidden_size, hidden_size * 2)
        self.value_range = range(hidden_size * 2, hidden_size * 3)
        self.output_range = range(hidden_size * 3, hidden_size * 4)
        
        # Set output dimension
        self.set_output_dim(hidden_size)
        
        # Build attention connectivity
        self._build_attention_connections()
    
    def _build_attention_connections(self):
        """Build attention mechanism connections."""
        # Query-Key connections
        for q in self.query_range:
            head_idx = q // self.head_size
            for k in range(
                self.key_range.start + head_idx * self.head_size,
                self.key_range.start + (head_idx + 1) * self.head_size
            ):
                self.add_synapse(q, k, 1)
        
        # Key-Value connections
        for k in self.key_range:
            head_idx = (k - self.key_range.start) // self.head_size
            for v in range(
                self.value_range.start + head_idx * self.head_size,
                self.value_range.start + (head_idx + 1) * self.head_size
            ):
                self.add_synapse(k, v, 1)
        
        # Value-Output connections
        for v in self.value_range:
            for o in self.output_range:
                self.add_synapse(v, o, 1)


def test_signal_wiring():
    """Test signal processing wiring pattern."""
    # Create wiring
    wiring = SignalWiring(
        input_size=1,
        num_bands=4,
        neurons_per_band=16,
        output_size=1
    )
    
    # Create model
    model = CfC(wiring=wiring)
    
    # Test forward pass
    batch_size = 32
    seq_length = 100
    x = mx.random.normal((batch_size, seq_length, 1))
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 1)
    
    # Check band connectivity
    for band_range in wiring.band_ranges:
        # Check internal connections
        for src in band_range:
            connected = False
            for dest in band_range:
                if src != dest and wiring.adjacency_matrix[src, dest] != 0:
                    connected = True
                    break
            assert connected, f"Neuron {src} has no connections within its band"


def test_robotics_wiring():
    """Test robotics wiring pattern."""
    # Create wiring
    wiring = RoboticsWiring(
        sensor_neurons=16,
        state_neurons=32,
        control_neurons=4,
        sensor_fanout=4,
        state_recurrent=3,
        control_fanin=4
    )
    
    # Create model
    model = LTC(wiring=wiring)
    
    # Test forward pass
    batch_size = 16
    seq_length = 50
    x = mx.random.normal((batch_size, seq_length, 16))  # Match sensor neurons
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 4)
    
    # Check layer connectivity
    # Sensor to state connections
    for src in wiring.sensor_range:
        connected = False
        for dest in wiring.state_range:
            if wiring.adjacency_matrix[src, dest] != 0:
                connected = True
                break
        assert connected, f"Sensor neuron {src} has no state connections"
    
    # State to control connections
    for dest in wiring.control_range:
        connected = False
        for src in wiring.state_range:
            if wiring.adjacency_matrix[src, dest] != 0:
                connected = True
                break
        assert connected, f"Control neuron {dest} has no state inputs"


def test_attention_wiring():
    """Test attention-based wiring pattern."""
    # Create wiring
    wiring = AttentionWiring(
        hidden_size=32,
        num_heads=4,
        ff_size=64
    )
    
    # Create model
    model = CfC(wiring=wiring)
    
    # Test forward pass
    batch_size = 8
    seq_length = 20
    x = mx.random.normal((batch_size, seq_length, 32))  # Match hidden size
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 32)
    
    # Check attention connectivity
    # Query-Key connections
    for q in wiring.query_range:
        head_idx = q // wiring.head_size
        connected = False
        for k in range(
            wiring.key_range.start + head_idx * wiring.head_size,
            wiring.key_range.start + (head_idx + 1) * wiring.head_size
        ):
            if wiring.adjacency_matrix[q, k] != 0:
                connected = True
                break
        assert connected, f"Query {q} has no key connections"
    
    # Value-Output connections
    for v in wiring.value_range:
        connected = False
        for o in wiring.output_range:
            if wiring.adjacency_matrix[v, o] != 0:
                connected = True
                break
        assert connected, f"Value {v} has no output connections"


def test_wiring_training():
    """Test training with advanced wiring patterns."""
    # Create models
    models = {
        'signal': CfC(SignalWiring(input_size=1, output_size=1)),
        'robotics': LTC(RoboticsWiring(16, 32, 4)),
        'attention': CfC(AttentionWiring(32, 4, 64))
    }
    
    # Test training step
    optimizer = nn.Adam(learning_rate=0.001)
    
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    for name, model in models.items():
        # Generate data
        batch_size = 16
        seq_length = 20
        input_size = model.cell.wiring.input_dim or 32  # Default to 32 if None
        x = mx.random.normal((batch_size, seq_length, input_size))
        y = mx.random.normal((batch_size, seq_length, model.cell.wiring.output_dim))
        
        # Single training step
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        
        assert not mx.isnan(loss)
        assert float(loss) > 0


def test_wiring_errors():
    """Test error handling in advanced wiring patterns."""
    # Test invalid signal wiring
    with pytest.raises(ValueError):
        SignalWiring(input_size=1, num_bands=0)  # Invalid number of bands
    
    # Test invalid robotics wiring
    with pytest.raises(ValueError):
        RoboticsWiring(
            sensor_neurons=16,
            state_neurons=32,
            control_neurons=4,
            sensor_fanout=64  # Too many connections
        )
    
    # Test invalid attention wiring
    with pytest.raises(ValueError):
        AttentionWiring(
            hidden_size=32,
            num_heads=5,  # Not divisible
            ff_size=64
        )


def test_wiring_serialization():
    """Test configuration serialization."""
    # Test signal wiring
    signal = SignalWiring(input_size=1, num_bands=4)
    config = signal.get_config()
    signal_restored = SignalWiring(**config)
    assert signal_restored.num_bands == signal.num_bands
    
    # Test robotics wiring
    robotics = RoboticsWiring(16, 32, 4)
    config = robotics.get_config()
    robotics_restored = RoboticsWiring(**config)
    assert robotics_restored.control_neurons == robotics.control_neurons
    
    # Test attention wiring
    attention = AttentionWiring(32, 4, 64)
    config = attention.get_config()
    attention_restored = AttentionWiring(**config)
    assert attention_restored.num_heads == attention.num_heads
