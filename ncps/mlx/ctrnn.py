"""Continuous-Time Recurrent Neural Network (CTRNN) implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mxu
from typing import Optional, Tuple, List, Union, Dict, Any

from .liquid_utils import get_activation


class CTRNNCell(nn.Module):
    """A Continuous-Time Recurrent Neural Network (CTRNN) cell.
    
    This cell follows MLX's unified class library design:
    1. Inherits from nn.Module for automatic parameter handling
    2. Uses underscore prefix for non-trainable attributes
    3. Direct attribute access for trainable parameters
    4. Uses centralized activation handling
    """
    
    def __init__(
        self,
        units: int,
        global_feedback: bool = False,
        activation: str = "tanh",
        cell_clip: Optional[float] = None,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """Initialize the CTRNN cell.
        
        Args:
            units: Number of units in the cell
            global_feedback: Whether to use global feedback
            activation: Activation function name to use
            cell_clip: Optional value to clip cell outputs
            epsilon: Small value for numerical stability
        """
        super().__init__()
        # Store non-trainable attributes
        self._units = units
        self._global_feedback = global_feedback
        self._activation_name = activation
        self._cell_clip = cell_clip
        self._epsilon = epsilon

        self._activation = get_activation(activation)
        self.input_linear: Optional[nn.Linear] = None
        self.recurrent_linear: Optional[nn.Linear] = None
        self._input_dim: Optional[int] = None
        self.tau = None

    @property
    def state_size(self):
        """Return the size of the cell state."""
        return self._units

    @property
    def output_size(self):
        """Return the size of the cell output."""
        return self._units

    def _ensure_parameters(self, input_dim: int) -> None:
        """Ensure projection layers match the current input dimension."""
        self._input_dim = input_dim
        if self.input_linear is None or self.input_linear.weight.shape[1] != input_dim:
            self.input_linear = nn.Linear(input_dim, self._units, bias=False)
            scale = mx.sqrt(2.0 / (input_dim + self._units))
            self.input_linear.weight = scale * mx.random.normal(self.input_linear.weight.shape)
        if self.recurrent_linear is None or self.recurrent_linear.weight.shape[1] != self._units:
            self.recurrent_linear = nn.Linear(self._units, self._units, bias=True)
            scale = mx.sqrt(2.0 / (2 * self._units))
            self.recurrent_linear.weight = scale * mx.random.normal(self.recurrent_linear.weight.shape)
            self.recurrent_linear.bias = 0.1 * mx.ones((self._units,))
        if self.tau is None or self.tau.shape[-1] != self._units:
            self.tau = 0.1 * mx.ones((self._units,))

    def __call__(
        self,
        inputs: mx.array,
        state: mx.array,
        time: float = 1.0,
        **kwargs
    ) -> Tuple[mx.array, mx.array]:
        """Process one time step.
        
        Args:
            inputs: Input tensor
            state: Previous state tensor
            time: Time step size
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, new_state) as MLX arrays
        """
        self._ensure_parameters(inputs.shape[-1])
        
        net = self.input_linear(inputs) + self.recurrent_linear(state)
        
        # Apply activation to get target state
        target_state = self._activation(net)
        
        # Update state using continuous-time dynamics with fixed time constant
        d_state = (-state + target_state) / (self.tau + self._epsilon)  # Add epsilon for stability
        output = state + time * d_state
        
        # Apply cell clipping if specified
        if self._cell_clip is not None:
            output = mx.clip(output, -self._cell_clip, self._cell_clip)
        
        return output, output

    def state_dict(self) -> Dict[str, Any]:
        parameters = mxu.tree_map(lambda arr: mx.array(arr), self.parameters())
        return {
            'config': {
                'units': self._units,
                'global_feedback': self._global_feedback,
                'activation': self._activation_name,
                'cell_clip': self._cell_clip,
                'epsilon': self._epsilon,
                'input_dim': self._input_dim,
            },
            'parameters': parameters,
            'buffers': {'tau': self.tau},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        config = state_dict.get('config', {})
        if config:
            self._units = config['units']
            self._global_feedback = config['global_feedback']
            self._activation_name = config['activation']
            self._activation = get_activation(self._activation_name)
            self._cell_clip = config['cell_clip']
            self._epsilon = config['epsilon']
            self._input_dim = config.get('input_dim', self._input_dim)
        buffers = state_dict.get('buffers', {})
        if 'tau' in buffers:
            self.tau = buffers['tau']
        params = state_dict.get('parameters', {})
        if params:
            input_dim = self._input_dim
            if input_dim is None:
                weight = params.get('input_linear.weight')
                if weight is not None:
                    input_dim = weight.shape[1]
                else:
                    input_dim = self._units
            self._ensure_parameters(input_dim)
        if params:
            self.update(params)
