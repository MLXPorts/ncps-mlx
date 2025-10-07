from ncps.activations.base import Activation

class ReLUBase(Activation):
    """Base class for ReLU activation function."""
    
    def __str__(self) -> str:
        return "ReLU"