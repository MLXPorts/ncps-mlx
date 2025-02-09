import mlx.nn as nn
from ncps.mini_keras.backend.common import name_scope
from ncps.mini_keras.backend.mlx import backend
from ncps.mini_keras.layers.layer import Layer

class MLXLayer(Layer, nn.Module):
    """MLX-specific layer base class that bridges Keras Layer with MLX Module.
    
    This class provides dual inheritance to get:
    1. MLX Module functionality for GPU acceleration
    2. Keras Layer functionality for the neural network framework
    
    Key features:
    - Name scope handling
    - MLX parameters tracking with GPU support
    - Keras Layer API compatibility
    - MLX-specific variable attributes
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        nn.Module.__init__(self)
        Layer.__init__(self, *args, **kwargs)
        self._supports_masking = False
        self._stateless_scope = None
        self._name_scope = None

    def __call__(self, *args, **kwargs):
        # Use Layer's __call__ to maintain Keras behavior
        return Layer.__call__(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        # Bridge to Keras' call method
        return self.call(*args, **kwargs)

    @property 
    def parameters(self):
        # Override nn.Module parameters to use Keras variable tracking
        params = {}
        for var in self.variables:
            params[var.name] = var.value
        return params

    def __setattr__(self, name, value):
        # Handle both MLX and Keras variable tracking
        if isinstance(value, backend.Variable):
            with name_scope.current_name_scope():
                value = value.copy()
        nn.Module.__setattr__(self, name, value)
