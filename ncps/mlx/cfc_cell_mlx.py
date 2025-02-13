"""Closed-form Continuous-time (CfC) cell implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from .base import LiquidCell
from .liquid_utils import get_activation
from .typing import InitializerCallable


class CfCCell(LiquidCell):
    """A Closed-form Continuous-time (CfC) cell."""
    
    def __init__(
        self,
        wiring,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        initializer: Optional[InitializerCallable] = None,
    ):
        """Initialize the CfC cell."""
        # Initialize base class
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            initializer=initializer,
        )
        
        # Validate and store mode
        self.mode = mode
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")
            
    def __call__(self, x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array]:
        """Process one step with the CfC cell.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            state: Previous state tensor of shape [batch_size, hidden_size]
            time: Time delta since last update
            
        Returns:
            Tuple of (output, new_state) tensors
        """
        # Build lazy parameters if not built
        if not hasattr(self, 'ff1_kernel'):
            self._build(x.shape[-1])
            
        # Get input dimensions
        batch_size = x.shape[0]
        
        # Combine input and state
        concat_input = mx.concatenate([x, state], axis=-1)
        
        # Apply backbone if present
        concat_input = self.apply_backbone(concat_input)
            
        # Apply main transformation
        ff1 = mx.matmul(concat_input, self.ff1_kernel) + self.ff1_bias
            
        if self.mode == "pure":
            if isinstance(time, mx.array):
                time = time[:, None]  # Add dimension for broadcasting
            new_state = (
                -self.A 
                * mx.exp(-time * (mx.abs(self.w_tau) + mx.abs(ff1))) 
                * ff1 
                + self.A
            )
        else:
            ff2 = mx.matmul(concat_input, self.ff2_kernel) + self.ff2_bias
                
            t_a = self.time_a(concat_input)
            t_b = self.time_b(concat_input)
            if isinstance(time, mx.array):
                time = time[:, None]  # Add dimension for broadcasting
            t_interp = nn.sigmoid(-t_a * time + t_b)
            
            if self.mode == "no_gate":
                new_state = ff1 + t_interp * ff2
            else:
                new_state = ff1 * (1.0 - t_interp) + t_interp * ff2
        
        # Project to output dimension if different from hidden size
        output = new_state
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
        
        # Get effective input dimension based on backbone
        if self.backbone is not None:
            input_dim = self.backbone_output_dim
        else:
            input_dim = self.input_size + self.hidden_size
                
        # Initialize main transformation weights
        self.ff1_kernel = self.initializer((input_dim, self.hidden_size))
        self.ff1_bias = mx.zeros((self.hidden_size,))
        
        if self.mode == "pure":
            self.w_tau = mx.zeros((1, self.hidden_size))
            self.A = mx.ones((1, self.hidden_size))
        else:
            # Initialize second transformation
            self.ff2_kernel = self.initializer((input_dim, self.hidden_size))
            self.ff2_bias = mx.zeros((self.hidden_size,))
            
            # Initialize time projection layers
            self.time_a = nn.Linear(input_dim, self.hidden_size)
            self.time_b = nn.Linear(input_dim, self.hidden_size)
            
            self.time_a.weight = self.initializer((self.hidden_size, input_dim))
            self.time_a.bias = mx.zeros((self.hidden_size,))
            self.time_b.weight = self.initializer((self.hidden_size, input_dim))
            self.time_b.bias = mx.zeros((self.hidden_size,))
            
        # Initialize output projection if needed
        if self.wiring.output_dim != self.hidden_size:
            self.output_kernel = self.initializer((self.hidden_size, self.wiring.output_dim))
            self.output_bias = mx.zeros((self.wiring.output_dim,))

    def state_dict(self) -> Dict[str, Any]:
        """Return the cell's state dictionary."""
        state = super().state_dict()
        state['mode'] = self.mode
        
        if hasattr(self, 'ff1_kernel'):
            state['ff1_kernel'] = self.ff1_kernel
            state['ff1_bias'] = self.ff1_bias
            
            if self.mode == 'pure':
                state['w_tau'] = self.w_tau
                state['A'] = self.A
            else:
                state['ff2_kernel'] = self.ff2_kernel
                state['ff2_bias'] = self.ff2_bias
                state['time_a'] = self.time_a.parameters()
                state['time_b'] = self.time_b.parameters()
                
            if self.wiring.output_dim != self.hidden_size:
                state['output_kernel'] = self.output_kernel
                state['output_bias'] = self.output_bias
                
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the cell's state from a dictionary."""
        super().load_state_dict(state_dict)
        self.mode = state_dict['mode']
        
        if 'ff1_kernel' in state_dict:
            self.ff1_kernel = state_dict['ff1_kernel']
            self.ff1_bias = state_dict['ff1_bias']
            
            if self.mode == 'pure':
                self.w_tau = state_dict['w_tau']
                self.A = state_dict['A']
            else:
                self.ff2_kernel = state_dict['ff2_kernel']
                self.ff2_bias = state_dict['ff2_bias']
                input_dim = self.ff1_kernel.shape[0]
                self.time_a = nn.Linear(input_dim, self.hidden_size)
                self.time_b = nn.Linear(input_dim, self.hidden_size)
                self.time_a.update(state_dict['time_a'])
                self.time_b.update(state_dict['time_b'])
                
            if 'output_kernel' in state_dict:
                self.output_kernel = state_dict['output_kernel']
                self.output_bias = state_dict['output_bias']
