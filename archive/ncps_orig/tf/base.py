# Copyright 2022-2025 The NCPS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base classes for TensorFlow implementation."""

import tensorflow as tf
from typing import List, Optional, Union

class LiquidCell(tf.keras.layers.Layer):
    """Base class for liquid neural network cells."""

    @property
    def output_size(self):
        """Return output size for RNN."""
        return self.wiring.output_dim

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.output_size)
    """Base class for liquid neural network cells.
    
    This class serves as the foundation for all liquid neural network cells,
    providing common functionality for backbone networks, dimension tracking,
    and state management.
    
    Args:
        wiring: Neural circuit wiring specification
        activation: Activation function name or callable
        backbone_units: List of backbone layer sizes
        backbone_layers: Number of backbone layers
        backbone_dropout: Backbone dropout rate
    """
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
    ):
        super().__init__()
        self.wiring = wiring
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.backbone_units = backbone_units or []
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        self.backbone_output_dim = None
        
    def build_backbone(self):
        """Build backbone network layers."""
        layers = []
        current_dim = self.input_size + self.hidden_size
        
        for i, units in enumerate(self.backbone_units):
            layers.append(tf.keras.layers.Dense(units))
            layers.append(tf.keras.layers.Activation(self.activation))
            if self.backbone_dropout > 0:
                layers.append(tf.keras.layers.Dropout(self.backbone_dropout))
        
        return tf.keras.Sequential(layers) if layers else None
    
    def build(self, input_shape):
        """Build cell parameters."""
        # Calculate dimensions
        self.input_size = input_shape[-1]
        self.hidden_size = self.wiring.units
        
        # Build backbone if specified
        if self.backbone_layers > 0:
            self.backbone = self.build_backbone()
            self.backbone_output_dim = self.backbone_units[-1]
        else:
            self.backbone_output_dim = self.input_size + self.hidden_size
            
        self.built = True
    
    @property
    def state_size(self):
        """Return state size for RNN."""
        return self.hidden_size
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Create initial state."""
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = self.dtype or tf.float32
            
        return [
            tf.zeros((batch_size, self.hidden_size), dtype=dtype)
        ]
    
    def call(self, inputs, states, training=None):
        """Process one step with the cell.
        
        Args:
            inputs: Input tensor
            states: List of state tensors
            training: Whether in training mode
            
        Returns:
            Tuple of (output tensor, list of new state tensors)
        """
        raise NotImplementedError()
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
        })
        return config