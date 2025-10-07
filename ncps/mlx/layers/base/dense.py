import mlx.core as mx
from typing import Any, Optional, Dict, Union, Tuple, List

from ncps.layers.base.dense import DenseBase
from ncps.mlx.layers.base.layer import MLXLayer
from ncps.mlx.activations import ReLU, Sigmoid, Tanh

class Dense(DenseBase, MLXLayer):
    """MLX implementation of the dense layer."""
    
    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            dtype=dtype,
            **kwargs
        )
        
        # Set MLX-specific activation
        if activation is not None:
            if activation == 'relu':
                self.activation = ReLU()
            elif activation == 'sigmoid':
                self.activation = Sigmoid()
            elif activation == 'tanh':
                self.activation = Tanh()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
    
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build the dense layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            dtype=self.dtype,
            name="kernel"
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                dtype=self.dtype,
                name="bias"
            )
            
        self.built = True
    
    def call(self, inputs: mx.array, training: Optional[bool] = None) -> mx.array:
        """Dense layer forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        outputs = mx.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs