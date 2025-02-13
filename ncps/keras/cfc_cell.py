"""Closed-form Continuous-time (CfC) cell implementation for Keras."""

import keras
from keras import ops, layers, activations
from typing import Optional, List, Dict, Any, Union, Tuple

from .activations import LeCunTanh
from .base import LiquidCell


@keras.saving.register_keras_serializable(package="ncps")
class CfCCell(LiquidCell):
    """A Closed-form Continuous-time (CfC) cell."""
    
    def __init__(
        self,
        wiring,
        mode: str = "default",
        activation: Union[str, layers.Layer] = "tanh",
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
        
        # Validate and store mode
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")
        self.mode = mode
        
        # Get activation function
        if activation == "lecun_tanh":
            self.activation = LeCunTanh()
            self.activation_name = "lecun_tanh"
        else:
            self.activation = activations.get(activation)
            self.activation_name = activation
        
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
                if isinstance(self.activation, layers.Layer):
                    layers_list.append(self.activation)
                else:
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
        
        # Initialize main transformation weights
        self.ff1_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="ff1_kernel"
        )
        self.ff1_bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="ff1_bias"
        )
        
        if self.mode == "pure":
            self._build_pure_mode()
        else:
            self._build_gated_mode(input_dim)
        
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
    
    def _build_pure_mode(self):
        """Build pure mode parameters."""
        self.w_tau = self.add_weight(
            shape=(1, self.units),
            initializer="zeros",
            name="w_tau"
        )
        self.A = self.add_weight(
            shape=(1, self.units),
            initializer="ones",
            name="A"
        )
    
    def _build_gated_mode(self, input_dim: int):
        """Build gated mode parameters."""
        self.ff2_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="ff2_kernel"
        )
        self.ff2_bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="ff2_bias"
        )
        
        # Initialize time projection layers
        self.time_a = layers.Dense(self.units, name="time_a")
        self.time_b = layers.Dense(self.units, name="time_b")
        
        # Build time projection layers
        self.time_a.build((None, input_dim))
        self.time_b.build((None, input_dim))
    
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
        
        # Apply transformations
        if self.mode == "pure":
            new_state = self._pure_step(x)
        else:
            new_state = self._gated_step(x)
        
        # Project output if needed
        if self.output_size != self.units:
            output = ops.matmul(new_state, self.output_kernel) + self.output_bias
        else:
            output = new_state
        
        return output, [new_state]
    
    def _pure_step(self, x):
        """Execute pure mode step."""
        ff1 = ops.matmul(x, self.ff1_kernel) + self.ff1_bias
        new_state = (
            -self.A 
            * ops.exp(-(ops.abs(self.w_tau) + ops.abs(ff1))) 
            * ff1 
            + self.A
        )
        return new_state
    
    def _gated_step(self, x):
        """Execute gated mode step."""
        ff1 = ops.matmul(x, self.ff1_kernel) + self.ff1_bias
        ff2 = ops.matmul(x, self.ff2_kernel) + self.ff2_bias
        
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = activations.sigmoid(-t_a + t_b)
        
        if self.mode == "no_gate":
            new_state = ff1 + t_interp * ff2
        else:
            new_state = ff1 * (1.0 - t_interp) + t_interp * ff2
        
        return new_state
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'wiring': self.wiring.get_config(),
            'mode': self.mode,
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
