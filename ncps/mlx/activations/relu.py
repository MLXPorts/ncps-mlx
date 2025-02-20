import mlx.core as mx
from ncps.activations.relu import ReLUBase

class ReLU(ReLUBase):
    """MLX implementation of ReLU activation function."""
    
    def __call__(self, x):
        """Apply ReLU activation using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The activated tensor
        """
        return mx.maximum(0, x)
    
    def gradient(self, x):
        """Calculate ReLU gradient using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The gradient tensor
        """
        return mx.where(x > 0, 1.0, 0.0)