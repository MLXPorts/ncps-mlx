from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, Tuple

class Layer(ABC):
    """Abstract base class for all layers.
    
    This serves as the foundation for all layer implementations across different
    backends (MLX, Keras, etc.). Each backend should provide concrete implementations
    of these methods.
    """
    
    def __init__(self, dtype: str = "float32", **kwargs):
        """Initialize the layer.
        
        Args:
            dtype: Data type for layer weights
            **kwargs: Additional keyword arguments
        """
        self.built = False
        self.trainable_weights: List[Any] = []
        self.non_trainable_weights: List[Any] = []
        self._name = kwargs.get('name', None)
        self.dtype = dtype
    
    @abstractmethod
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build the layer weights.
        
        This method should be implemented by subclasses to create their weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        pass
    
    @abstractmethod
    def __call__(self, inputs: Any, training: Optional[bool] = None, **kwargs) -> Any:
        """Layer forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def add_weight(
        self,
        shape: Tuple[int, ...],
        initializer: str = "glorot_uniform",
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[str] = None
    ) -> Any:
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
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Dictionary containing the layer configuration
        """
        return {
            'name': self._name,
            'dtype': self.dtype
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Layer':
        """Create layer from configuration.
        
        Args:
            config: Layer configuration dictionary
            
        Returns:
            Created layer instance
        """
        return cls(**config)