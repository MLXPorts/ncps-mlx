from ncps.activations.base import Activation

class TanhBase(Activation):
    """Base class for Tanh activation function."""
    
    def __str__(self) -> str:
        return "Tanh"