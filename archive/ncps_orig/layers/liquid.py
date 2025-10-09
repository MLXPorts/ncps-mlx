"""Liquid neuron implementations."""

import numpy as np
from ncps import ops
from typing import Optional, List, Union, Dict, Any

from .rnn import RNNCell
from .layer import Dense


class LiquidCell(RNNCell):
    """Base class for liquid neurons."""
    
    def __init__(
        self,
        units: int,
        activation: str = "tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(units, dtype=dtype, **kwargs)
        
        # Get activation function
        self.activation_name = activation
        self.activation_fn = getattr(ops, activation)
        
        # Store backbone config
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        
        # For tracking dimensions
        self.input_dim = None
        self.backbone_output_dim = None
    
    def build(self, input_shape):
        """Build the cell."""
        # Get input dimension
        self.input_dim = input_shape[-1]
        
        # Build backbone if needed
        if self.backbone_layers > 0 and self.backbone_units is not None:
            self.backbone = []
            current_dim = self.input_dim + self.units  # Combined input and state
            
            for i in range(self.backbone_layers):
                # Add dense layer
                self.backbone.append(
                    Dense(
                        self.backbone_units,
                        activation=self.activation_name,
                        dtype=self.dtype,
                        name=f"backbone_{i}"
                    )
                )
                
                # Add dropout if needed
                if self.backbone_dropout > 0:
                    self.backbone.append(
                        lambda x, training: ops.dropout(
                            x,
                            rate=self.backbone_dropout,
                            training=training
                        )
                    )
                
                current_dim = self.backbone_units
                
            self.backbone_output_dim = self.backbone_units
        else:
            self.backbone_output_dim = None
            
        self.built = True
    
    def apply_backbone(
        self,
        x: np.ndarray,
        training: Optional[bool] = None
    ) -> np.ndarray:
        """Apply backbone network if present."""
        if self.backbone is not None:
            for layer in self.backbone:
                if callable(layer):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x
    
    def call(
        self,
        inputs: np.ndarray,
        states: List[np.ndarray],
        training: Optional[bool] = None,
        **kwargs
    ):
        """Process one timestep."""
        # Get current state
        h = states[0]
        
        # Combine inputs and state
        features = ops.concatenate([inputs, h], axis=-1)
        
        # Apply backbone if present
        if self.backbone_output_dim is not None:
            # Apply backbone with dropout
            features = self.apply_backbone(features, training=training)
            # Apply activation
            output = self.activation_fn(features)
            # Return same size state as input state
            new_state = output[:, :self.units]
            # Return processed features as output
            return output, [new_state]
        else:
            # Apply activation
            output = self.activation_fn(features)
            # Return same size state as input state
            new_state = output[:, :self.units]
            # Return concatenated features as output
            return features, [new_state]
    
    def get_config(self):
        """Get cell configuration."""
        config = super().get_config()
        config.update({
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
        })
        return config


class CfCCell(LiquidCell):
    """Closed-form Continuous-time cell."""
    
    def __init__(
        self,
        units: int,
        mode: str = "pure",
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(units, dtype=dtype, **kwargs)
        self.mode = mode
        
        # Initialize mode-specific weights
        self.w_tau = None
        self.A = None
        self.gate_kernel = None
        self.gate_bias = None
    
    def build(self, input_shape):
        """Build the cell."""
        super().build(input_shape)
        
        if self.mode == "pure":
            # Time constant weights
            self.w_tau = self.add_weight(
                shape=(1, self.units),
                initializer="zeros",
                dtype=self.dtype,
                name="w_tau"
            )
            # Amplitude weights
            self.A = self.add_weight(
                shape=(1, self.units),
                initializer="ones",
                dtype=self.dtype,
                name="A"
            )
        else:
            # Gate weights
            self.gate_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer="glorot_uniform",
                dtype=self.dtype,
                name="gate_kernel"
            )
            self.gate_bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                dtype=self.dtype,
                name="gate_bias"
            )
    
    def call(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        states: List[np.ndarray],
        training: Optional[bool] = None,
        **kwargs
    ):
        """Process one timestep."""
        # Get time input
        if isinstance(inputs, (list, tuple)):
            x, t = inputs
            if isinstance(t, (int, float)):
                t = ops.ones_like(x[:, :1], dtype=self.dtype) * t
            t = ops.reshape(t, [-1, 1])
        else:
            x = inputs
            t = ops.ones_like(x[:, :1], dtype=self.dtype)
        
        # Get current state
        h = states[0]
        
        # Combine inputs and state
        features = ops.concatenate([x, h], axis=-1)
        
        # Apply backbone if present
        if self.backbone_output_dim is not None:
            # Apply backbone with dropout
            features = self.apply_backbone(features, training=training)
        
        # Apply mode-specific update
        if self.mode == "pure":
            # Pure mode with direct ODE solution
            state = -self.A * ops.exp(-t * (ops.abs(self.w_tau) + ops.abs(h))) * h + self.A
        elif self.mode == "gated":
            # Gated mode with interpolation
            gate = ops.matmul(h, self.gate_kernel)
            gate = gate + self.gate_bias
            gate = ops.sigmoid(-t * gate)
            state = features[:, :self.units] * (1.0 - gate) + gate * h
        else:
            # No-gate mode with simple update
            gate = ops.matmul(h, self.gate_kernel) + self.gate_bias
            state = features[:, :self.units] + t * ops.tanh(gate)
        
        # Return processed features as output and new state
        return features, [state]
    
    def get_config(self):
        """Get cell configuration."""
        config = super().get_config()
        config.update({
            'mode': self.mode
        })
        return config