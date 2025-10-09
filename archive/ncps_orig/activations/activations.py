from abc import ABC, abstractmethod
from typing import Any, Literal

ActivationType = Literal['relu', 'sigmoid', 'tanh']

class Activation(ABC):
    """Abstract base class for activation functions.
    
    This class defines the interface that all activation implementations must follow.
    Each backend (MLX, Keras, etc.) should provide a concrete implementation
    of these methods.
    
    Args:
        activation_type: The type of activation function to use
    """
    
    def __init__(self, activation_type: ActivationType):
        """Initialize the activation function.
        
        Args:
            activation_type: Type of activation function ('relu', 'sigmoid', 'tanh')
        """
        self.activation_type = activation_type
    
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
