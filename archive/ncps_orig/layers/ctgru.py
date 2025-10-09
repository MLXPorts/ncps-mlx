"""Continuous-time Gated Recurrent Unit (CT-GRU) cell implementation."""

import keras
from keras import ops, layers
from typing import Optional, List, Any, Union, Tuple

from .base import BackboneLayerCell


@keras.saving.register_keras_serializable(package="ncps")
class CTGRUCell(BackboneLayerCell):
    """A Continuous-time Gated Recurrent Unit cell.
    
    This implements a continuous-time version of the GRU,
    with time-dependent update gates.
    
    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        recurrent_activation: Activation function for recurrent step.
        backbone_units: Number of units in backbone layers.
        backbone_layers: Number of backbone layers.
        backbone_dropout: Dropout rate in backbone layers.
        **kwargs: Additional keyword arguments for the base layer.
    """
    
    def __init__(
        self,
        units: int,
        activation: Union[str, layers.Layer] = "tanh",
        recurrent_activation: Union[str, layers.Layer] = "sigmoid",
        backbone_units: Optional[int] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(units, **kwargs)
        
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone_fn = None
        
        # Calculate backbone dimensions
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units
        else:
            self.backbone_output_dim = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the cell weights.
        
        Args:
            input_shape: Tuple of integers, the input shape.
        """
        super().build(input_shape)
        
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build backbone if needed
        if self.backbone_layers > 0 and self.backbone_units:
            backbone_layers = []
            for i in range(self.backbone_layers):
                backbone_layers.append(
                    layers.Dense(
                        self.backbone_units,
                        self.activation,
                        name=f"backbone{i}"
                    )
                )
                if self.backbone_dropout > 0:
                    backbone_layers.append(
                        layers.Dropout(self.backbone_dropout)
                    )
            
            self.backbone_fn = keras.Sequential(backbone_layers)
            self.backbone_fn.build((None, self.units + input_dim))
            cat_shape = self.backbone_units
        else:
            cat_shape = self.units + input_dim
        
        # Input transformation weights
        self.kernel = self.add_weight(
            shape=(cat_shape, 3 * self.units),  # [z, r, h]
            initializer="glorot_uniform",
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(3 * self.units,),
            initializer="zeros",
            name="bias"
        )
        
        # Time scaling network
        self.time_kernel = layers.Dense(
            self.units,
            name="time_kernel"
        )
        self.time_kernel.build((None, cat_shape))
    
    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        states: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """Process one timestep.
        
        Args:
            inputs: Input tensor or list of [input, time] tensors.
            states: List of state tensors.
            training: Whether in training mode.
            
        Returns:
            Tuple of (output tensor, list of new state tensors).
        """
        # Handle time input
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = ops.reshape(t, [-1, 1])
        else:
            t = 1.0
        
        # Get current state
        h_prev = states[0]
        
        # Combine inputs and state
        x = layers.Concatenate()([inputs, h_prev])
        
        # Apply backbone if present
        if self.backbone_fn is not None:
            x = self.backbone_fn(x, training=training)
        
        # Compute gates and candidate
        gates = ops.matmul(x, self.kernel) + self.bias
        z, r, h_tilde = ops.split(gates, 3, axis=-1)
        
        # Apply activations
        z = self.recurrent_activation(z)  # Update gate
        r = self.recurrent_activation(r)  # Reset gate
        h_tilde = self.activation(h_tilde)  # Candidate activation
        
        # Compute time scaling
        tau = ops.exp(self.time_kernel(x))
        
        # Update state
        h_new = h_prev + t * (
            -z * h_prev +  # Decay current state
            (1 - z) * (r * h_tilde)  # Add new information
        ) / tau
        
        return h_new, [h_new]
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "activation": keras.activations.serialize(self.activation),
            "recurrent_activation": keras.activations.serialize(self.recurrent_activation),
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout
        })
        return config