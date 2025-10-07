"""Continuous-Time Recurrent Neural Network (CTRNN) implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union

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
        self.built = False

        # Get activation function from centralized utilities
        self._activation = get_activation(activation)

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
        
        scale = mx.sqrt(2.0 / (2 * self._units))
        self.recurrent_kernel = scale * mx.random.normal((self._units, self._units))
        
        # Initialize bias with small positive values for stability
        self.bias = 0.1 * mx.ones((self._units,))
        
        # Initialize fixed time constant
        self.tau = 0.1 * mx.ones((self._units,))  # Smaller time constant for faster dynamics
        
        self.built = True

    def __call__(
        self,
        inputs: mx.array,
        state: Union[mx.array, List[mx.array]],
        time: float = 1.0,
        **kwargs
    ) -> Tuple[mx.array, List[mx.array]]:
        """Process one time step.
        
        Args:
            inputs: Input tensor
            state: Previous state tensor or list containing state tensor
            time: Time step size
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, new_states)
        """
        if not self.built:
            self.build(inputs.shape)

        # Extract state from list if needed
        prev_output = state[0] if isinstance(state, list) else state
        
        # Compute net input with proper MLX operations
        net = mx.matmul(inputs, self.kernel)
        net = net + mx.matmul(prev_output, self.recurrent_kernel)
        net = net + self.bias
        
        # Apply activation to get target state
        target_state = self._activation(net)
        
        # Update state using continuous-time dynamics with fixed time constant
        d_state = (-prev_output + target_state) / (self.tau + self._epsilon)  # Add epsilon for stability
        output = prev_output + time * d_state
        
        # Apply cell clipping if specified
        if self._cell_clip is not None:
            output = mx.clip(output, -self._cell_clip, self._cell_clip)
        
        return output, [output]