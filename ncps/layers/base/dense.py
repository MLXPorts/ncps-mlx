from typing import Any, Optional, Dict, Union, Tuple

from ncps.layers.base.layer import Layer

class DenseBase(Layer):
    """Base class for dense (fully connected) layers.
    
    This class defines the interface for dense layer implementations across
    different backends. Each backend should provide a concrete implementation
    of the abstract methods.
    
    Args:
        units: Number of output units
        activation: Optional activation function to use
        use_bias: Whether to include a bias term
        dtype: Data type for layer weights
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = None  # Set by backend implementation
        self.use_bias = use_bias
        
        # Set by build
        self.kernel = None
        self.bias = None
    
    @abstractmethod
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build the dense layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        pass
    
    @abstractmethod
    def __call__(self, inputs: Any, training: Optional[bool] = None, **kwargs) -> Any:
        """Dense layer forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation_name,
            'use_bias': self.use_bias
        })
        return config