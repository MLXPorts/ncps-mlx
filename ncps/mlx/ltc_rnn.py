import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

from .ltc import LTC

class LTCRNN(nn.Module):
    """Compatibility wrapper around :class:`LTC` with state round-tripping."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        sparsity_mask: Optional[mx.array] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

        self.layer = LTC(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            return_sequences=True,
            return_state=True,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )

    def __call__(
        self,
        x: mx.array,
        initial_states: Optional[List[mx.array]] = None,
        time_delta: Optional[Union[float, mx.array]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        squeeze_output = False
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 0)
            squeeze_output = True

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
        return {
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'bias': self.bias,
                'bidirectional': self.bidirectional,
                'activation': self.activation,
                'backbone_units': self.backbone_units,
                'backbone_layers': self.backbone_layers,
                'backbone_dropout': self.backbone_dropout,
            },
            'layer': self.layer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        config = state_dict.get('config', {})
        if config:
            self.input_size = config.get('input_size', self.input_size)
            self.hidden_size = config.get('hidden_size', self.hidden_size)
            self.num_layers = config.get('num_layers', self.num_layers)
            self.bias = config.get('bias', self.bias)
            self.bidirectional = config.get('bidirectional', self.bidirectional)
            self.activation = config.get('activation', self.activation)
            self.backbone_units = config.get('backbone_units', self.backbone_units)
            self.backbone_layers = config.get('backbone_layers', self.backbone_layers)
            self.backbone_dropout = config.get('backbone_dropout', self.backbone_dropout)
        self.layer.load_state_dict(state_dict['layer'])
