"""Utility functions and mixins for liquid neural networks."""

from typing import Optional, Tuple, List, Union, Dict, Any, Callable

import mlx.core as mx
import mlx.nn as nn


class TimeAwareMixin:
    """Mixin class for handling time-aware updates in liquid neurons."""
    
    def process_time_delta(
        self,
        time_delta: Optional[Union[float, mx.array]],
        batch_size: int,
        seq_len: int
    ) -> mx.array:
        """Process time delta input into a usable format.
        
        Args:
            time_delta: Time delta as float or array
            batch_size: Batch size for broadcasting
            seq_len: Sequence length for broadcasting
            
        Returns:
            Processed time delta array
        """
        if time_delta is None:
            return mx.ones((batch_size, seq_len, 1))
        elif isinstance(time_delta, (int, float)):
            return mx.full((batch_size, seq_len, 1), time_delta)
        else:
            # Ensure time_delta has shape [batch_size, seq_len, 1]
            if time_delta.ndim == 1:
                time_delta = mx.expand_dims(time_delta, axis=0)
            if time_delta.ndim == 2:
                time_delta = mx.expand_dims(time_delta, axis=-1)
            return time_delta


class BackboneMixin:
    """Mixin class for handling backbone layers in liquid neurons."""
    
    def build_backbone(
        self,
        input_size: int,
        backbone_units: int,
        backbone_layers: int,
        backbone_dropout: float,
        initializer: Optional[Any] = None
    ) -> List[nn.Linear]:
        """Build backbone layers for feature extraction.
        
        Args:
            input_size: Size of input features
            backbone_units: Number of units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone layers
            initializer: Optional weight initializer
            
        Returns:
            List of backbone layers
        """
        layers = []
        current_size = input_size
        
        for i in range(backbone_layers):
            layer = nn.Linear(
                current_size,
                backbone_units,
                bias=True
            )
            if initializer:
                layer.weight = initializer((backbone_units, current_size))
                layer.bias = mx.zeros((backbone_units,))
            layers.append(layer)
            current_size = backbone_units
            
        return layers

    def apply_backbone(
        self,
        x: mx.array,
        backbone_layers: List[nn.Linear],
        activation: Callable,
        dropout: float,
        training: bool = True
    ) -> mx.array:
        """Apply backbone layers to input.
        
        Args:
            x: Input tensor
            backbone_layers: List of backbone layers
            activation: Activation function to use
            dropout: Dropout rate
            training: Whether in training mode
            
        Returns:
            Processed tensor
        """
        h = x
        for layer in backbone_layers:
            h = activation(layer(h))
            if training and dropout > 0:
                h = nn.dropout(h, dropout)
        return h


def lecun_tanh(x: mx.array) -> mx.array:
    """LeCun's tanh activation function.
    
    Applies the scaled tanh activation: 1.7159 * tanh(0.666 * x)
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return 1.7159 * mx.tanh(0.666 * x)


def sigmoid(x: mx.array) -> mx.array:
    """Sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return 1.0 / (1.0 + mx.exp(-x))


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
        "tanh": mx.tanh,
        "relu": nn.relu,
        "gelu": nn.gelu,
        "sigmoid": sigmoid,
        "identity": lambda x: x,
    }
    
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Valid options are: {list(activations.keys())}"
        )
    
    return activations[name]
