"""Closed-form Continuous-time (CfC) RNN implementation for Keras."""

import keras
from typing import Optional, List, Dict, Any, Union

from .cfc_cell import CfCCell


@keras.saving.register_keras_serializable(package="ncps")
class CfC(keras.Model):
    """A Closed-form Continuous-time (CfC) RNN layer.
    
    This layer wraps the CfCCell with sequence processing capabilities
    and bidirectional support.
    
    Args:
        wiring: Neural circuit wiring
        mode: Operation mode ('default', 'pure', 'no_gate')
        activation: Activation function
        backbone_units: Backbone layer sizes
        backbone_layers: Number of backbone layers
        backbone_dropout: Backbone dropout rate
        return_sequences: Whether to return full sequence
        return_state: Whether to return final state
        go_backwards: Whether to process sequence backwards
        stateful: Whether to reuse state between batches
        unroll: Whether to unroll the RNN
        zero_output_for_mask: Whether to use zeros for masked timesteps
    """
    
    def __init__(
        self,
        wiring,
        mode: str = "default",
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        return_sequences: bool = True,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        zero_output_for_mask: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Create CfC cell
        self.cell = CfCCell(
            wiring=wiring,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        
        # Store configuration
        self.mode = mode
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.wiring = wiring
        self.activation = activation
        
        # Store RNN configuration
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.zero_output_for_mask = zero_output_for_mask
        
        # Create RNN layer
        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            zero_output_for_mask=zero_output_for_mask,
        )
    
    def build(self, input_shape):
        """Build the layer."""
        # Build RNN
        self.rnn.build(input_shape)
        self.built = True
    
    def call(self, inputs, training=None, mask=None, initial_state=None):
        """Process input sequence.
        
        Args:
            inputs: Input tensor or list of tensors
            training: Whether in training mode
            mask: Optional mask tensor
            initial_state: Optional initial state
            
        Returns:
            Output tensor or list of tensors
        """
        # Handle inputs
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]  # Ignore time inputs for now
        
        # Call RNN
        return self.rnn(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        
        # Add CfC-specific config
        config.update({
            'wiring': self.wiring.get_config(),
            'mode': self.mode,
            'activation': self.activation,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'unroll': self.unroll,
            'zero_output_for_mask': self.zero_output_for_mask,
        })
        
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create from configuration."""
        # Extract wiring configuration
        wiring_config = config.pop('wiring')
        
        # Import wiring class dynamically
        import importlib
        wirings_module = importlib.import_module('ncps.keras.wirings')
        wiring_class = getattr(wirings_module, wiring_config['class_name'])
        
        # Create wiring
        wiring = wiring_class.from_config(wiring_config['config'])
        config['wiring'] = wiring
        
        # Remove cell config if present
        config.pop('cell', None)
        
        return cls(**config)
    
    @property
    def state_size(self):
        """Get state size."""
        return self.cell.state_size
    
    @property
    def output_size(self):
        """Get output size."""
        return self.cell.output_size
