"""Basic layer system for liquid neural networks."""

import numpy as np
from ncps import ops
from typing import Optional, Dict, Any, List, Union, Tuple


class Layer:
    """Base layer class with minimal functionality needed for liquid neurons."""
    
    def __init__(self, dtype: str = "float32", **kwargs):
        self.built = False
        self.trainable_weights = []
        self.non_trainable_weights = []
        self._name = kwargs.get('name', None)
        self.dtype = dtype
    
    def build(self, input_shape):
        """Create layer weights."""
        self.built = True
    
    def add_weight(
        self,
        shape: Tuple[int, ...],
        initializer: str = "glorot_uniform",
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[str] = None
    ) -> np.ndarray:
        """Add a weight variable to the layer."""
        dtype = dtype or self.dtype
        
        # Handle initializers
        if initializer == "glorot_uniform":
            fan_in = np.prod(shape[:-1])
            fan_out = shape[-1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            weight = np.random.uniform(-limit, limit, size=shape)
        elif initializer == "zeros":
            weight = np.zeros(shape)
        elif initializer == "ones":
            weight = np.ones(shape)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
            
        weight = ops.convert_to_tensor(weight, dtype=dtype)
        
        if trainable:
            self.trainable_weights.append(weight)
        else:
            self.non_trainable_weights.append(weight)
            
        return weight
    
    def __call__(self, inputs, *args, **kwargs):
        """Layer call wrapper."""
        if not self.built:
            # Handle list/tuple inputs
            if isinstance(inputs, (list, tuple)):
                input_shape = inputs[0].shape
            else:
                input_shape = inputs.shape
            self.build(input_shape)
        return self.call(inputs, *args, **kwargs)
    
    def call(self, inputs, training=None, **kwargs):
        """Layer forward pass."""
        raise NotImplementedError()
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        return {
            'name': self._name,
            'dtype': self.dtype
        }
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


class Dense(Layer):
    """Basic dense layer."""
    
    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation = getattr(ops, activation) if activation else None
        self.use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            dtype=self.dtype,
            name="kernel"
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                dtype=self.dtype,
                name="bias"
            )
            
        self.built = True
        
    def call(self, inputs, training=None):
        outputs = ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation.__name__ if self.activation else None,
            'use_bias': self.use_bias
        })
        return config


class Sequential:
    """Sequential container of layers."""
    
    def __init__(self, layers=None):
        self.layers = layers or []
        
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
        
    def __call__(self, inputs, training=None, **kwargs):
        """Forward pass through all layers."""
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training, **kwargs)
        return x
    
    def get_weights(self) -> List[Any]:
        """Get all layer weights."""
        weights = []
        for layer in self.layers:
            weights.extend(layer.trainable_weights)
            weights.extend(layer.non_trainable_weights)
        return weights
    
    def set_weights(self, weights: List[Any]):
        """Set all layer weights."""
        weight_idx = 0
        for layer in self.layers:
            num_weights = (
                len(layer.trainable_weights) + 
                len(layer.non_trainable_weights)
            )
            layer_weights = weights[weight_idx:weight_idx + num_weights]
            
            trainable_count = len(layer.trainable_weights)
            layer.trainable_weights = layer_weights[:trainable_count]
            layer.non_trainable_weights = layer_weights[trainable_count:]
            
            weight_idx += num_weights