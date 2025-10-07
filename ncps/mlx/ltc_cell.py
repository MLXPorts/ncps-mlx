"""Liquid Time-Constant (LTC) cell implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from .base import LiquidCell  
from .liquid_utils import get_activation
from .typing import InitializerCallable


class LTCCell(LiquidCell):
    """Liquid Time-Constant (LTC) cell implementation."""
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        initializer: Optional[InitializerCallable] = None,
    ):
        """Initialize the LTC cell."""
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            initializer=initializer,
        )
        
        # Initialize dimensions
        self.input_size = wiring.input_dim if wiring.input_dim is not None else 0
        self.hidden_size = wiring.units
        
        # Calculate backbone dimensions
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units[-1]
        else:
            # If no backbone, concatenated input will be used directly
            self.backbone_output_dim = self.input_size + self.hidden_size
            
    def __call__(self, x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array]:
        """Forward pass of the LTC cell.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            state: Previous cell state of shape [batch_size, hidden_size]
            time: Time step size for ODE integration
            
        Returns:
            Tuple of:
                - Output tensor of shape [batch_size, hidden_size]
                - New state tensor of shape [batch_size, hidden_size]
        """
        # Build lazy parameters if not built
        if not hasattr(self, 'kernel'):
            self._build(x.shape[-1])
            
        # Get input dimensions
        batch_size = x.shape[0]
        
        # Combine input and state
        concat_input = mx.concatenate([x, state], axis=-1)
        
        # Apply backbone if present
        if self.backbone is not None:
            concat_input = self.backbone(concat_input)
            
        # Compute delta term
        d = mx.matmul(concat_input, self.kernel) + self.bias
            
        # Compute time constants
        tau = mx.exp(self.tau_kernel(concat_input))
        
        # Update state using time constant
        if isinstance(time, mx.array):
            time = time[:, None]  # Add dimension for broadcasting
        new_state = state + time * (-state + d) / tau
        
        # Apply activation to state for output
        output = self.activation(new_state)
        
        # Project to output dimension if different from hidden size
        if self.wiring.output_dim != self.hidden_size:
            output = mx.matmul(output, self.output_kernel) + self.output_bias
        
        return output, new_state
        
    def _build(self, input_dim: int) -> None:
        """Build the cell parameters.
        
        Args:
            input_dim: Dimension of the input features
        """
        # Set input dimension
        self.input_size = input_dim
        
        # Calculate concatenated input dimension
        cat_dim = self.input_size + self.hidden_size
        
        # Get effective input dimension based on backbone
        if self.backbone is not None:
            input_dim = self.backbone_output_dim
        else:
            input_dim = cat_dim
                
        # Initialize main transformation weights
        self.kernel = self.initializer((input_dim, self.hidden_size))
        self.bias = mx.zeros((self.hidden_size,))
        
        # Initialize time constant network
        self.tau_kernel = nn.Linear(input_dim, self.hidden_size)
        self.tau_kernel.weight = self.initializer((self.hidden_size, input_dim))
        self.tau_kernel.bias = mx.zeros((self.hidden_size,))
        
        # Initialize output projection if needed
        if self.wiring.output_dim != self.hidden_size:
            self.output_kernel = self.initializer((self.hidden_size, self.wiring.output_dim))
            self.output_bias = mx.zeros((self.wiring.output_dim,))
        
    def state_dict(self) -> Dict[str, Any]:
        """Return the cell's state dictionary."""
        state = super().state_dict()
        
        if hasattr(self, 'kernel'):
            state['kernel'] = self.kernel
            state['bias'] = self.bias
            state['tau_kernel'] = self.tau_kernel.parameters()
            
            if self.wiring.output_dim != self.hidden_size:
                state['output_kernel'] = self.output_kernel
                state['output_bias'] = self.output_bias
            
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the cell's state from a dictionary."""
        super().load_state_dict(state_dict)
        
        if 'kernel' in state_dict:
            self.kernel = state_dict['kernel']
            self.bias = state_dict['bias']
            self.tau_kernel = nn.Linear(self.kernel.shape[0], self.hidden_size)
            self.tau_kernel.update(state_dict['tau_kernel'])
            
            if 'output_kernel' in state_dict:
                self.output_kernel = state_dict['output_kernel']
                self.output_bias = state_dict['output_bias']
