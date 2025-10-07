"""Closed-form Continuous-time (CfC) cell implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from .base import LiquidCell
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
        self._effective_input_dim: Optional[int] = None
            
    def _ensure_parameters(self, input_dim: int) -> None:
        """Build or rebuild trainable parameters to match the input dimension."""
        self.input_size = input_dim
        concat_dim = self.input_size + self.hidden_size
        self.backbone_input_dim = concat_dim
        
        if self.backbone_layers > 0 and self.backbone_units:
            if self._backbone_built_for != concat_dim:
                self.backbone_input_dim = concat_dim
                self.build_backbone()
                self._backbone_built_for = concat_dim
            effective_dim = self.backbone_output_dim
        else:
            self._backbone_layers = []
            self.backbone = None
            self.backbone_output_dim = concat_dim
            effective_dim = concat_dim
        
        rebuild_ff1 = (
            not hasattr(self, 'ff1') or
            self.ff1.weight.shape[1] != effective_dim or
            self.ff1.weight.shape[0] != self.hidden_size
        )
        if rebuild_ff1:
            self.ff1 = nn.Linear(effective_dim, self.hidden_size)
            self.ff1.weight = self.initializer((self.hidden_size, effective_dim))
            self.ff1.bias = mx.zeros((self.hidden_size,))
        
        if self.mode == "pure":
            if not hasattr(self, 'w_tau') or self.w_tau.shape[-1] != self.hidden_size:
                self.w_tau = mx.zeros((1, self.hidden_size))
                self.A = mx.ones((1, self.hidden_size))
            # Remove gated parameters if they exist
            if hasattr(self, 'ff2'):
                del self.ff2
            if hasattr(self, 'time_a'):
                del self.time_a
            if hasattr(self, 'time_b'):
                del self.time_b
        else:
            rebuild_ff2 = (
                not hasattr(self, 'ff2') or
                self.ff2.weight.shape[1] != effective_dim or
                self.ff2.weight.shape[0] != self.hidden_size
            )
            if rebuild_ff2:
                self.ff2 = nn.Linear(effective_dim, self.hidden_size)
                self.ff2.weight = self.initializer((self.hidden_size, effective_dim))
                self.ff2.bias = mx.zeros((self.hidden_size,))
            rebuild_time = (
                not hasattr(self, 'time_a') or
                self.time_a.weight.shape[1] != effective_dim
            )
            if rebuild_time:
                self.time_a = nn.Linear(effective_dim, self.hidden_size)
                self.time_a.weight = self.initializer((self.hidden_size, effective_dim))
                self.time_a.bias = mx.zeros((self.hidden_size,))
                self.time_b = nn.Linear(effective_dim, self.hidden_size)
                self.time_b.weight = self.initializer((self.hidden_size, effective_dim))
                self.time_b.bias = mx.zeros((self.hidden_size,))
            if hasattr(self, 'w_tau'):
                del self.w_tau
            if hasattr(self, 'A'):
                del self.A
        
        if self.wiring.output_dim != self.hidden_size:
            need_output = (
                not hasattr(self, 'output_proj') or
                self.output_proj.weight.shape[0] != self.wiring.output_dim or
                self.output_proj.weight.shape[1] != self.hidden_size
            )
            if need_output:
                self.output_proj = nn.Linear(self.hidden_size, self.wiring.output_dim)
                self.output_proj.weight = self.initializer((self.wiring.output_dim, self.hidden_size))
                self.output_proj.bias = mx.zeros((self.wiring.output_dim,))
        elif hasattr(self, 'output_proj'):
            del self.output_proj

    def __call__(self, x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array]:
        """Process one step with the CfC cell.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            state: Previous state tensor of shape [batch_size, hidden_size]
            time: Time delta since last update
            
        Returns:
            Tuple of (output, new_state) tensors
        """
        # Ensure parameters are built for the current input size
        self._ensure_parameters(x.shape[-1])
        
        # Combine input and state
        concat_input = mx.concatenate([x, state], axis=-1)
        
        # Apply backbone if present
        concat_input = self.apply_backbone(concat_input, training=self.training)
            
        # Apply main transformation
        ff1 = self.ff1(concat_input)
            
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
            ff2 = self.ff2(concat_input)
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
            output = self.output_proj(output)
                
        return output, new_state
        
    def _build_from_config(self, config: Dict[str, Any]) -> None:
        """Ensure modules exist when restoring from state."""
        self.mode = config.get('mode', self.mode)
        wiring_cfg = config.get('wiring', {})
        input_dim = config.get('input_dim')
        if input_dim is None:
            input_dim = wiring_cfg.get('input_dim', self.input_size)
        if input_dim is not None:
            self._ensure_parameters(input_dim)
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state['config']['mode'] = self.mode
        return state
