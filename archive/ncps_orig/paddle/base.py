"""Base classes for PaddlePaddle implementation."""

import paddle
import paddle.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Union


class LiquidCell(nn.Layer):
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
        self.units = wiring.units
        self._state_size = self.units
        self._output_size = wiring.output_dim or wiring.units
        self._input_size = wiring.input_dim if wiring.input_dim is not None else 0
        
        # Get activation function
        if isinstance(activation, str):
            if activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "sigmoid":
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        self.activation_name = activation if isinstance(activation, str) else activation.__name__
        
        # Process backbone units
        if backbone_units is None:
            backbone_units = []
        elif isinstance(backbone_units, int):
            backbone_units = [backbone_units] * backbone_layers
        elif len(backbone_units) == 1 and backbone_layers > 1:
            backbone_units = backbone_units * backbone_layers
            
        # Store backbone configuration
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        
        # Calculate backbone dimensions
        self.backbone_input_dim = self._input_size + self.units
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units[-1]
        else:
            self.backbone_output_dim = self.backbone_input_dim
    
    @property
    def state_size(self):
        """Get size of cell state."""
        return self._state_size

    @property
    def output_size(self):
        """Get size of cell output."""
        return self._output_size

    @property
    def input_size(self):
        """Get size of cell input."""
        return self._input_size

    def build_backbone(self):
        """Build backbone network layers."""
        # Clear any existing backbone
        self.backbone = None
        
        # Only build if needed
        if not self.backbone_layers or not self.backbone_units:
            return
        
        layers = []
        current_dim = self._input_size + self.units
        
        # Build layers
        for i, units in enumerate(self.backbone_units):
            # Add linear layer
            layers.append(nn.Linear(current_dim, units))
            
            # Add activation layer
            layers.append(self.activation)
            
            # Add dropout if needed
            if self.backbone_dropout > 0:
                layers.append(nn.Dropout(self.backbone_dropout))
            
            current_dim = units
            
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, inputs, states):
        """Process one step with the cell.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            states: Previous state tensors
            
        Returns:
            Tuple of (output tensor, new_states)
        """
        raise NotImplementedError()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from configuration."""
        # Extract wiring configuration
        wiring_config = config.pop('wiring')
        
        # Import wiring class dynamically
        import importlib
        wirings_module = importlib.import_module('ncps.wirings')
        wiring_class = getattr(wirings_module, wiring_config['class_name'])
        
        # Create wiring
        wiring = wiring_class.from_config(wiring_config['config'])
        config['wiring'] = wiring
        
        return cls(**config)