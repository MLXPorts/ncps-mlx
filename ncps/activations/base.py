from abc import ABC, abstractmethod
from typing import Any

class Activation(ABC):
    """Abstract base class for all activation functions."""
    
    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """Apply the activation function.
        
        Args:
            x: Input tensor/array
            
        Returns:
            The result of applying the activation function to the input
        """
        pass
    
    @abstractmethod
    def gradient(self, x: Any) -> Any:
        """Calculate the gradient of the activation function.
        
        Args:
            x: Input tensor/array
            
        Returns:
            The gradient of the activation function with respect to the input
        """
        pass