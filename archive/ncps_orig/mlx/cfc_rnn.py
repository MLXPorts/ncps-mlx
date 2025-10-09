import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

from .cfc import CfC

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
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

        self.layer = CfC(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            return_sequences=True,
            return_state=True,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
    
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
        
        outputs, states = self.layer(
            x,
            initial_states=initial_states,
            time_delta=time_delta,
        )
        if squeeze_output:
            outputs = mx.squeeze(outputs, axis=0)
            states = [mx.squeeze(state, axis=0) for state in states]
        return outputs, states
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the layer's state dictionary."""
        return {
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'bias': self.bias,
                'bidirectional': self.bidirectional,
                'mode': self.mode,
                'activation': self.activation,
                'backbone_units': self.backbone_units,
                'backbone_layers': self.backbone_layers,
                'backbone_dropout': self.backbone_dropout,
            },
            'layer': self.layer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the layer's state from a dictionary."""
        config = state_dict.get('config', {})
        if config:
            self.input_size = config.get('input_size', self.input_size)
            self.hidden_size = config.get('hidden_size', self.hidden_size)
            self.num_layers = config.get('num_layers', self.num_layers)
            self.bias = config.get('bias', self.bias)
            self.bidirectional = config.get('bidirectional', self.bidirectional)
            self.mode = config.get('mode', self.mode)
            self.activation = config.get('activation', self.activation)
            self.backbone_units = config.get('backbone_units', self.backbone_units)
            self.backbone_layers = config.get('backbone_layers', self.backbone_layers)
            self.backbone_dropout = config.get('backbone_dropout', self.backbone_dropout)
        self.layer.load_state_dict(state_dict['layer'])
