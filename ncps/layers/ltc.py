"""Linear Time-invariant Continuous-time (LTC) cell implementation."""

import keras
from keras import ops, layers
from typing import Optional, List, Any, Union, Tuple

from .base import BackboneLayerCell


@keras.saving.register_keras_serializable(package="ncps")
class LTCCell(BackboneLayerCell):
    """A Linear Time-invariant Continuous-time (LTC) cell.
    
    This implements the LTC with time-constant based processing and
    optional backbone network for enhanced representation.
    
    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        backbone_units: Number of units in backbone layers.
        backbone_layers: Number of backbone layers.
        backbone_dropout: Dropout rate in backbone layers.
        **kwargs: Additional keyword arguments for the base layer.
    """
    
    def __init__(
        self,
        units: int,
        activation: Union[str, layers.Layer] = "tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(units, **kwargs)
        
        self.activation = keras.activations.get(activation)
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
        
        # Initialize main transformation weights
        self.kernel = self.add_weight(
            shape=(cat_shape, self.units),
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
        self.tau_kernel.build((None, cat_shape))
    
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
        state = states[0]
        
        # Combine inputs and state
        x = layers.Concatenate()([inputs, state])
        
        # Apply backbone if present
        if self.backbone_fn is not None:
            x = self.backbone_fn(x, training=training)
        
        # Compute delta term
        d = ops.matmul(x, self.kernel) + self.bias
        
        # Compute time constants
        tau = ops.exp(self.tau_kernel(x))
        
        # Update state using time constant
        new_state = state + t * (-state + d) / tau
        
        # Apply activation
        output = self.activation(new_state)
        
        return output, [new_state]
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "activation": keras.activations.serialize(self.activation),
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout
        })
        return config