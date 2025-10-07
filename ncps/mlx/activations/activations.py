import mlx.core as mx
from ncps.activations.activations import Activation, ActivationType

class MLXActivation(Activation):
    """MLX implementation of activation functions."""
    
    def __call__(self, x):
        """Apply the activation function using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The activated tensor
        """
        if self.activation_type == 'relu':
            return mx.maximum(0, x)
        elif self.activation_type == 'sigmoid':
            return mx.sigmoid(x)
        elif self.activation_type == 'tanh':
            return mx.tanh(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
    
    def gradient(self, x):
        """Calculate the gradient of the activation function using MLX.
        
        Args:
            x: Input tensor
            
        Returns:
            The gradient tensor
        """
        if self.activation_type == 'relu':
            return mx.where(x > 0, 1.0, 0.0)
        elif self.activation_type == 'sigmoid':
            sig = mx.sigmoid(x)
            return sig * (1 - sig)
        elif self.activation_type == 'tanh':
            tanh_x = mx.tanh(x)
            return 1 - mx.square(tanh_x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")