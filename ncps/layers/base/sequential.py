from typing import Any, List, Optional, Dict, Union, Tuple

from ncps.layers.base.layer import Layer

class SequentialBase(Layer):
    """Base class for sequential container of layers.
    
    This class defines the interface for sequential model implementations across
    different backends. Each backend should provide a concrete implementation
    of the abstract methods.
    
    Args:
        layers: Optional list of layers to add to the model
        dtype: Data type for layer weights
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        layers: Optional[List[Layer]] = None,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.layers = layers or []
    
    def add(self, layer: Layer) -> None:
        """Add a layer to the model.
        
        Args:
            layer: Layer instance to add
        """
        self.layers.append(layer)
    
    @abstractmethod
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build all layers in the model.
        
        Args:
            input_shape: Shape of the input tensor
        """
        pass
    
    @abstractmethod
    def __call__(self, inputs: Any, training: Optional[bool] = None, **kwargs) -> Any:
        """Sequential model forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> List[Any]:
        """Get weights from all layers.
        
        Returns:
            List of all weight tensors
        """
        pass
    
    @abstractmethod
    def set_weights(self, weights: List[Any]) -> None:
        """Set weights for all layers.
        
        Args:
            weights: List of weight tensors to set
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        # Note: Layers are not included in config as they need
        # backend-specific serialization
        return config