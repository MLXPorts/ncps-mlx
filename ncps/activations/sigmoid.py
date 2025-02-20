from ncps.activations.base import Activation

class SigmoidBase(Activation):
    """Base class for Sigmoid activation function."""
    
    def __str__(self) -> str:
        return "Sigmoid"