import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

from .cfc_cell_mlx import CfCCell

class CfCRNN(nn.Module):
    """An RNN layer using CfC (Closed-form Continuous-time) cells.
    
    The input is a sequence of shape NLD or LD where:
    - N is the optional batch dimension
    - L is the sequence length
    - D is the input's feature dimension
    
    For each element along the sequence length axis, this layer applies the CfC
    transformation with optional bidirectional processing and time-aware updates.
    
    The hidden state has shape NH or H (per direction), depending on whether the
    input is batched or not. Returns the hidden state at each time step, of shape
    NLH or LH (doubled for bidirectional).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        sparsity_mask: Optional[mx.array] = None,
    ):
        """Initialize the CfCRNN layer.

        Args:
            input_size: Dimension of the input features (D).
            hidden_size: Dimension of the hidden state (H).
            num_layers: Number of stacked CfC layers.
            bias: Whether to use bias in the cells.
            bidirectional: Whether to process sequences in both directions.
            mode: CfC mode ("default", "pure", or "no_gate").
            activation: Activation function to use.
            backbone_units: Number of units in backbone layers.
            backbone_layers: Number of backbone layers.
            backbone_dropout: Dropout rate for backbone layers.
            sparsity_mask: Optional sparsity mask for weights.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        
        # Create cells for each layer and direction
        self.forward_layers = []
        self.backward_layers = []
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            # Forward cell
            self.forward_layers.append(CfCCell(
                units=hidden_size,
                mode=mode,
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                sparsity_mask=sparsity_mask
            ))
            
            # Backward cell if bidirectional
            if bidirectional:
                self.backward_layers.append(CfCCell(
                    units=hidden_size,
                    mode=mode,
                    activation=activation,
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_dropout=backbone_dropout,
                    sparsity_mask=sparsity_mask
                ))
    
    def __call__(
        self, 
        x: mx.array, 
        initial_states: Optional[List[mx.array]] = None,
        time_delta: Optional[Union[float, mx.array]] = None
    ) -> Tuple[mx.array, List[mx.array]]:
        """Process input sequence through the CfC layers.

        Args:
            x: Input tensor of shape [batch, seq_len, input_size] or [seq_len, input_size]
            initial_states: Optional list of initial states for each layer and direction
            time_delta: Optional time steps between sequence elements

        Returns:
            Tuple of:
            - Output tensor of shape [batch, seq_len, hidden_size * directions] or
              [seq_len, hidden_size * directions]
            - List of final states for each layer and direction
        """
        # Handle non-batched input
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len, _ = x.shape
        
        # Initialize states if not provided
        if initial_states is None:
            initial_states = []
            for _ in range(self.num_layers):
                initial_states.append(mx.zeros((batch_size, self.hidden_size)))
                if self.bidirectional:
                    initial_states.append(mx.zeros((batch_size, self.hidden_size)))
        
        # Process each layer
        current_input = x
        final_states = []
        
        for layer in range(self.num_layers):
            forward_cell = self.forward_layers[layer]
            backward_cell = self.backward_layers[layer] if self.bidirectional else None
            
            # Forward pass
            forward_states = []
            state = initial_states[layer * (2 if self.bidirectional else 1)]
            
            for t in range(seq_len):
                dt = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta
                output, state = forward_cell(current_input[:, t], state, time=dt)
                forward_states.append(output)
            
            forward_output = mx.stack(forward_states, axis=1)
            final_states.append(state)
            
            # Backward pass if bidirectional
            if self.bidirectional:
                backward_states = []
                state = initial_states[layer * 2 + 1]
                
                for t in range(seq_len - 1, -1, -1):
                    dt = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta
                    output, state = backward_cell(current_input[:, t], state, time=dt)
                    backward_states.append(output)
                
                backward_output = mx.stack(backward_states[::-1], axis=1)
                final_states.append(state)
                
                # Combine forward and backward outputs
                current_input = mx.concatenate([forward_output, backward_output], axis=-1)
            else:
                current_input = forward_output
        
        # Handle non-batched output
        if squeeze_output:
            current_input = mx.squeeze(current_input, axis=0)
            final_states = [mx.squeeze(state, axis=0) for state in final_states]
        
        return current_input, final_states
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the layer's state dictionary."""
        state = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bias': self.bias,
            'bidirectional': self.bidirectional,
            'forward_layers': [cell.state_dict() for cell in self.forward_layers],
        }
        if self.bidirectional:
            state['backward_layers'] = [cell.state_dict() for cell in self.backward_layers]
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the layer's state from a dictionary."""
        self.input_size = state_dict['input_size']
        self.hidden_size = state_dict['hidden_size']
        self.num_layers = state_dict['num_layers']
        self.bias = state_dict['bias']
        self.bidirectional = state_dict['bidirectional']
        
        for idx, cell_state in enumerate(state_dict['forward_layers']):
            self.forward_layers[idx].load_state_dict(cell_state)
            
        if self.bidirectional:
            for idx, cell_state in enumerate(state_dict['backward_layers']):
                self.backward_layers[idx].load_state_dict(cell_state)