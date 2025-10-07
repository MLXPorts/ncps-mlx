"""Continuous-Time Gated Recurrent Unit (CTGRU) implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Union

from .liquid_utils import get_activation, sigmoid


class CTGRUCell(nn.Module):
    """A Continuous-Time Gated Recurrent Unit (CTGRU) cell.
    
    This cell follows MLX's unified class library design:
    1. Inherits from nn.Module for automatic parameter handling
    2. Uses underscore prefix for non-trainable attributes
    3. Direct attribute access for trainable parameters
    4. Uses centralized activation handling
    """
    
    def __init__(
        self,
        units: int,
        cell_clip: Optional[float] = None,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """Initialize the CTGRU cell.
        
        Args:
            units: Number of units in the cell
            cell_clip: Optional value to clip cell outputs
            epsilon: Small value for numerical stability
        """
        super().__init__()
        # Store non-trainable attributes
        self._units = units
        self._cell_clip = cell_clip
        self._epsilon = epsilon
        self.built = False

        # Get activation functions from centralized utilities
        self._tanh = get_activation("tanh")

    @property
    def state_size(self):
        """Return the size of the cell state."""
        return self._units

    @property
    def output_size(self):
        """Return the size of the cell output."""
        return self._units

    def build(self, input_shape: Tuple[int, ...]):
        """Build the cell parameters.
        
        Args:
            input_shape: Shape of the input tensor
        """
        input_dim = input_shape[-1]
        
        # Initialize weights with proper MLX operations using Xavier initialization
        scale = mx.sqrt(2.0 / (input_dim + self._units))
        self.kernel = scale * mx.random.normal((input_dim, self._units))
        self.recurrent_kernel = scale * mx.random.normal((self._units, self._units))
        self.bias = 0.1 * mx.ones((self._units,))
        
        # Initialize time constant
        self.tau = 0.1 * mx.ones((self._units,))  # Smaller time constant for faster dynamics
        
        self.built = True

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
        if not self.built:
            self.build(inputs.shape)
        
        # Compute gates with proper MLX operations
        net = mx.matmul(inputs, self.kernel)
        net = net + mx.matmul(state, self.recurrent_kernel)
        net = net + self.bias
        
        # Split activation for different purposes using centralized functions
        z = sigmoid(net)  # Update gate
        r = sigmoid(net)  # Reset gate
        c = self._tanh(net)  # Candidate state
        
        # Compute target state
        target_state = (1 - z) * state + z * (r * c)
        
        # Update state using continuous-time dynamics with fixed time constant
        d_state = (-state + target_state) / (self.tau + self._epsilon)  # Add epsilon for stability
        output = state + time * d_state
        
        # Apply cell clipping if specified
        if self._cell_clip is not None:
            output = mx.clip(output, -self._cell_clip, self._cell_clip)
        
        return output, output