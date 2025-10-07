import mlx.core as mx
from typing import Any, Optional, Dict, List, Union, Tuple

from ncps.layers.base.layer import Layer

class MLXLayer(Layer):
    """MLX implementation of the base layer class."""
    
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.built = True
    
    def __call__(self, inputs: Any, training: Optional[bool] = None, **kwargs) -> Any:
        """Layer forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        if not self.built:
            self.build(inputs.shape)
        return self.call(inputs, training=training, **kwargs)
    
    def call(self, inputs: Any, training: Optional[bool] = None, **kwargs) -> Any:
        """Layer forward pass implementation.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        return inputs
    
    def add_weight(
        self,
        shape: Tuple[int, ...],
        initializer: str = "glorot_uniform",
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[str] = None
    ) -> mx.array:
        """Add a weight variable to the layer.
        
        Args:
            shape: Shape of the weight tensor
            initializer: Weight initialization method
            trainable: Whether the weight is trainable
            name: Optional name for the weight
            dtype: Optional dtype for the weight
            
        Returns:
            The created weight tensor
        """
        dtype = dtype or self.dtype
        
        # Handle initializers
        if initializer == "glorot_uniform":
            fan_in = mx.prod(mx.array(shape[:-1]))
            fan_out = shape[-1]
            limit = mx.sqrt(6 / (fan_in + fan_out))
            weight = mx.random.uniform(low=-limit, high=limit, shape=shape)
        elif initializer == "zeros":
            weight = mx.zeros(shape)
        elif initializer == "ones":
            weight = mx.ones(shape)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
        
        if trainable:
            self.trainable_weights.append(weight)
        else:
            self.non_trainable_weights.append(weight)
            
        return weight