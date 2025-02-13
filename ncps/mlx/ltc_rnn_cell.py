import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict, Any

from .ltc_cell import LTCCell as BaseLTCCell

class LTCRNNCell(nn.Module):
    """An LTC cell that follows MLX's RNNCell pattern.
    
    This cell can be used as a drop-in replacement for MLX's RNNCell in custom
    architectures. It follows the same interface pattern as MLX's built-in RNN cells.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        sparsity_mask: Optional[mx.array] = None,
    ):
        """Initialize the LTCRNNCell.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            bias: Whether to use bias
            activation: Activation function to use
            backbone_units: Number of units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone layers
            sparsity_mask: Optional sparsity mask for weights
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.cell = BaseLTCCell(
            units=hidden_size,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            sparsity_mask=sparsity_mask,
        )
        
    def __call__(
        self, 
        x: mx.array, 
        state: Optional[mx.array] = None,
        time_delta: Optional[float] = None
    ) -> Tuple[mx.array, mx.array]:
        """Process one time step.
        
        Args:
            x: Input tensor of shape [batch, input_size] or [input_size]
            state: Hidden state of shape [batch, hidden_size] or [hidden_size]
            time_delta: Optional time step size
            
        Returns:
            Tuple of (output, new_state) of shapes:
            - [batch, hidden_size] or [hidden_size]
            - [batch, hidden_size] or [hidden_size]
        """
        # Handle non-batched input
        if len(x.shape) == 1:
            x = mx.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = x.shape[0]
        
        # Initialize state if not provided
        if state is None:
            state = mx.zeros((batch_size, self.hidden_size))
        elif len(state.shape) == 1:
            state = mx.expand_dims(state, 0)
            
        # Process step
        output, new_state = self.cell(x, state, time=time_delta if time_delta is not None else 1.0)
        
        # Handle non-batched output
        if squeeze_output:
            output = mx.squeeze(output, axis=0)
            new_state = mx.squeeze(new_state, axis=0)
            
        return output, new_state
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the cell's state dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'bias': self.bias,
            'cell': self.cell.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the cell's state from a dictionary."""
        self.input_size = state_dict['input_size']
        self.hidden_size = state_dict['hidden_size']
        self.bias = state_dict['bias']
        self.cell.load_state_dict(state_dict['cell'])