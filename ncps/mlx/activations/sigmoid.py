import mlx.core as mx
from ncps.activations.sigmoid import SigmoidBase

class Sigmoid(SigmoidBase):
    """MLX implementation of Sigmoid activation function."""
    
    def __call__(self, x):
        """Apply sigmoid activation using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The activated tensor
        """
        return mx.sigmoid(x)
    
    def gradient(self, x):
        """Calculate sigmoid gradient using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The gradient tensor
        """
        sig = mx.sigmoid(x)
        return sig * (1 - sig)