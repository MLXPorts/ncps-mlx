"""Utility functions and mixins for liquid neural networks."""

import keras
from keras import ops, layers
from typing import Optional, List, Union, Dict, Any, Callable


class TimeAwareMixin:
    """Mixin class for handling time-aware updates in liquid neurons."""
    
    def process_time_delta(
        self,
        time_delta: Optional[Union[float, keras.KerasTensor]],
        batch_size: int,
        seq_len: Optional[int] = None
    ) -> keras.KerasTensor:
        """Process time delta input into a usable format.
        
        Args:
            time_delta: Time delta as float or tensor
            batch_size: Batch size for broadcasting
            seq_len: Optional sequence length for RNN processing
            
        Returns:
            Processed time delta tensor
        """
        if time_delta is None:
            shape = (batch_size, seq_len, 1) if seq_len else (batch_size, 1)
            return ops.ones(shape)
            
        if isinstance(time_delta, (int, float)):
            shape = (batch_size, seq_len, 1) if seq_len else (batch_size, 1)
            return ops.full(shape, time_delta)
            
        # Handle tensor input
        if len(time_delta.shape) == 1:
            time_delta = ops.reshape(time_delta, (batch_size, 1))
        if len(time_delta.shape) == 2 and seq_len:
            time_delta = ops.expand_dims(time_delta, axis=-1)
            
        return time_delta


class BackboneMixin:
    """Mixin class for handling backbone layers in liquid neurons."""
    
    def build_backbone(
        self,
        input_size: int,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        activation: Union[str, Callable],
    ) -> List[layers.Layer]:
        """Build backbone layers for feature extraction.
        
        Args:
            input_size: Size of input features
            backbone_units: Number of units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone layers
            activation: Activation function to use
            
        Returns:
            List of backbone layers
        """
        backbone = []
        current_size = input_size
        
        for i in range(backbone_layers):
            # Add dense layer
            backbone.append(
                layers.Dense(
                    backbone_units,
                    activation=activation,
                    name=f"backbone_{i}"
                )
            )
            
            # Add dropout if needed
            if backbone_dropout > 0:
                backbone.append(
                    layers.Dropout(backbone_dropout)
                )
            
            current_size = backbone_units
            
        return backbone
    
    def apply_backbone(
        self,
        x: keras.KerasTensor,
        backbone_layers: List[layers.Layer],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply backbone layers to input.
        
        Args:
            x: Input tensor
            backbone_layers: List of backbone layers
            training: Whether in training mode
            
        Returns:
            Processed tensor
        """
        h = x
        for layer in backbone_layers:
            if isinstance(layer, layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)
        return h


def lecun_tanh(x: keras.KerasTensor) -> keras.KerasTensor:
    """LeCun's tanh activation function.
    
    Applies the scaled tanh activation: 1.7159 * tanh(0.666 * x)
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return 1.7159 * ops.tanh(0.666 * x)


def get_activation(name: str) -> Callable:
    """Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
        
    Raises:
        ValueError: If activation name is unknown
    """
    activations = {
        "lecun_tanh": lecun_tanh,
        "tanh": ops.tanh,
        "relu": ops.relu,
        "gelu": ops.gelu,
        "sigmoid": ops.sigmoid,
        "linear": lambda x: x,
    }
    
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Valid options are: {list(activations.keys())}"
        )
    
    return activations[name]


def ensure_time_dim(x: keras.KerasTensor) -> keras.KerasTensor:
    """Ensure tensor has time dimension.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with time dimension
    """
    if len(x.shape) == 2:
        return ops.expand_dims(x, axis=1)
    return x


def broadcast_to_batch(
    x: keras.KerasTensor,
    batch_size: int
) -> keras.KerasTensor:
    """Broadcast tensor to batch dimension.
    
    Args:
        x: Input tensor
        batch_size: Target batch size
        
    Returns:
        Broadcasted tensor
    """
    if len(x.shape) == 1:
        x = ops.expand_dims(x, axis=0)
    return ops.broadcast_to(x, (batch_size,) + x.shape[1:])