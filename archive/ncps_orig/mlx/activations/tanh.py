import mlx.core as mx
from ncps.activations.tanh import TanhBase

class Tanh(TanhBase):
    """MLX implementation of Tanh activation function."""
    
    def __call__(self, x):
        """Apply tanh activation using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The activated tensor
        """
        return mx.tanh(x)
    
    def gradient(self, x):
        """Calculate tanh gradient using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The gradient tensor
        """
        tanh_x = mx.tanh(x)
        return 1 - mx.square(tanh_x)