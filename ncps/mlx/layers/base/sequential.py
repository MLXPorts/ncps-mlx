import mlx.core as mx
from typing import Any, List, Optional, Dict, Union, Tuple

from ncps.layers.base.sequential import SequentialBase
from ncps.mlx.layers.base.layer import MLXLayer

class Sequential(SequentialBase, MLXLayer):
    """MLX implementation of the sequential container."""
    
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build all layers in the model.
        
        Args:
            input_shape: Shape of the input tensor
        """
        x_shape = input_shape
        for layer in self.layers:
            layer.build(x_shape)
            # Update shape for next layer based on current layer's output
            if hasattr(layer, 'units'):
                x_shape = (*x_shape[:-1], layer.units)
        self.built = True
    
    def call(self, inputs: mx.array, training: Optional[bool] = None, **kwargs) -> mx.array:
        """Sequential model forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training, **kwargs)
        return x
    
    def get_weights(self) -> List[mx.array]:
        """Get weights from all layers.
        
        Returns:
            List of all weight tensors
        """
        weights = []
        for layer in self.layers:
            weights.extend(layer.trainable_weights)
            weights.extend(layer.non_trainable_weights)
        return weights
    
    def set_weights(self, weights: List[mx.array]) -> None:
        """Set weights for all layers.
        
        Args:
            weights: List of weight tensors to set
        """
        weight_idx = 0
        for layer in self.layers:
            num_weights = (
                len(layer.trainable_weights) + 
                len(layer.non_trainable_weights)
            )
            layer_weights = weights[weight_idx:weight_idx + num_weights]
            
            trainable_count = len(layer.trainable_weights)
            layer.trainable_weights = layer_weights[:trainable_count]
            layer.non_trainable_weights = layer_weights[trainable_count:]
            
            weight_idx += num_weights