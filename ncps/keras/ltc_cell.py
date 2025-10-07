"""Liquid Time-Constant (LTC) cell implementation for Keras."""

import keras
from keras import ops, layers, activations
from typing import Optional, List, Dict, Any, Union, Tuple

from .base import LiquidCell


@keras.saving.register_keras_serializable(package="ncps")
class LTCCell(LiquidCell):
    """A Liquid Time-Constant (LTC) cell."""
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.wiring = wiring
        self.units = wiring.units
        self.state_size = self.units
        self.output_size = wiring.output_dim or wiring.units
        self.input_size = wiring.input_dim if wiring.input_dim is not None else 0
        
        # Get activation function
        self.activation_name = activation
        self.activation = activations.get(activation)
        
        # Process backbone units
        if backbone_units is None:
            backbone_units = []
        elif isinstance(backbone_units, int):
            backbone_units = [backbone_units] * backbone_layers
        elif len(backbone_units) == 1 and backbone_layers > 1:
            backbone_units = backbone_units * backbone_layers
            
        # Store backbone configuration
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        
        # Calculate backbone dimensions
        self.backbone_input_dim = self.input_size + self.units
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units[-1]
        else:
            self.backbone_output_dim = self.backbone_input_dim
    
    def build_backbone(self):
        """Build backbone network layers."""
        # Clear any existing backbone
        self.backbone = None
        
        # Only build if needed
        if not self.backbone_layers or not self.backbone_units:
            return
        
        layers_list = []
        current_dim = self.backbone_input_dim
        
        # Build layers
        for i, units in enumerate(self.backbone_units):
            # Add linear layer
            layers_list.append(layers.Dense(units))
            
            # Add activation and dropout except for last layer
            if i < len(self.backbone_units) - 1:
                layers_list.append(layers.Activation(self.activation))
                if self.backbone_dropout > 0:
                    layers_list.append(layers.Dropout(self.backbone_dropout))
            
            current_dim = units
            
        self.backbone = keras.Sequential(layers_list)
        
        # Build backbone with correct input shape
        self.backbone.build((None, self.backbone_input_dim))
    
    def build(self, input_shape):
        """Build cell parameters."""
        # Set input dimension
        input_dim = input_shape[-1]
        self.input_size = input_dim
        self.backbone_input_dim = self.input_size + self.units
        
        # Build backbone if needed
        self.build_backbone()
        
        # Get effective input dimension
        if self.backbone is not None:
            input_dim = self.backbone_output_dim
        else:
            input_dim = self.input_size + self.units
        
        # Initialize transformation weights
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="bias"
        )
        
        # Initialize time constant network
        self.tau_kernel = layers.Dense(
            self.units,
            name="tau_kernel"
        )
        self.tau_kernel.build((None, input_dim))
        
        # Initialize output projection if needed
        if self.output_size != self.units:
            self.output_kernel = self.add_weight(
                shape=(self.units, self.output_size),
                initializer="glorot_uniform",
                name="output_kernel"
            )
            self.output_bias = self.add_weight(
                shape=(self.output_size,),
                initializer="zeros",
                name="output_bias"
            )
        
        self.built = True
    
    def call(self, inputs, states, training=None):
        """Process one step with the cell.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            states: Previous state tensors
            training: Whether in training mode
            
        Returns:
            Tuple of (output, new_states)
        """
        # Get current state
        state = states[0]
        
        # Process input
        x = layers.concatenate([inputs, state])
        if self.backbone is not None:
            x = self.backbone(x, training=training)
        
        # Compute delta term
        d = ops.matmul(x, self.kernel) + self.bias
        
        # Compute time constants
        tau = ops.exp(self.tau_kernel(x))
        
        # Update state
        new_state = state + (-state + d) / tau
        
        # Apply activation
        new_state = self.activation(new_state)
        
        # Project output if needed
        if self.output_size != self.units:
            output = ops.matmul(new_state, self.output_kernel) + self.output_bias
        else:
            output = new_state
        
        return output, [new_state]
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
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
        
        return cls(**config)
