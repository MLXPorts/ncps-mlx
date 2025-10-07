"""Base classes for MLX neural circuit implementations."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mxu
from typing import Optional, Tuple, List, Dict, Any, Union

from .typing import TensorOrFloat, InitializerCallable
from .liquid_utils import get_activation


class LiquidCell(nn.Module):
    """Base class for liquid neural network cells."""
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        initializer: Optional[InitializerCallable] = None,
    ):
        super().__init__()
        self.wiring = wiring
        self.units = wiring.units
        self.hidden_size = wiring.units
        self.input_size = wiring.input_dim if wiring.input_dim is not None else 0
        
        # Get activation function
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        # Default to Glorot uniform initialization
        if initializer is None:
            def glorot_uniform(shape):
                limit = mx.sqrt(6 / sum(shape))
                return mx.random.uniform(low=-limit, high=limit, shape=shape)
            self.initializer = glorot_uniform
        else:
            self.initializer = initializer

        # Process backbone units
        if backbone_units is None:
            backbone_units = []
        elif isinstance(backbone_units, int):
            backbone_units = [backbone_units] * backbone_layers
        elif len(backbone_units) == 1 and backbone_layers > 1:
            backbone_units = backbone_units * backbone_layers
            
        # Store backbone configuration
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        self._backbone_layers: List[nn.Linear] = []
        self._backbone_built_for: Optional[int] = None
        
        # Calculate backbone dimensions
        self.backbone_input_dim = self.input_size + self.hidden_size
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units[-1]
            self.build_backbone()
        else:
            self.backbone_output_dim = self.backbone_input_dim

    def build_backbone(self):
        """Build backbone network layers."""
        layers = []
        current_dim = self.backbone_input_dim
        
        self._backbone_layers = []
        for units in self.backbone_units:
            layer = nn.Linear(current_dim, units)
            if callable(self.initializer):
                layer.weight = self.initializer((units, current_dim))
                layer.bias = mx.zeros((units,))
            self._backbone_layers.append(layer)
            current_dim = units
        self.backbone = nn.Sequential(*self._backbone_layers) if self._backbone_layers else None
        self._backbone_built_for = self.backbone_input_dim
        self.backbone_output_dim = current_dim if self._backbone_layers else self.backbone_input_dim

    def apply_backbone(self, x: mx.array, training: bool = True) -> mx.array:
        """Apply backbone network if present."""
        if self.backbone is not None:
            # Get input shape
            batch_size = x.shape[0]
            
            # Ensure input has correct shape
            if x.shape[-1] != self.backbone_input_dim:
                raise ValueError(f"Expected input dimension {self.backbone_input_dim}, got {x.shape[-1]}")
            
            h = x
            for idx, layer in enumerate(self._backbone_layers):
                h = layer(h)
                if idx < len(self._backbone_layers) - 1:
                    h = self.activation(h)
                    if training and self.backbone_dropout > 0:
                        h = nn.dropout(h, self.backbone_dropout)
            x = h
            
        return x

    def __call__(self, x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array]:
        """Process one step with the cell.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            state: Previous state tensor of shape [batch_size, hidden_size] 
            time: Time delta since last update
            
        Returns:
            Tuple of (output, new_state) tensors
            
        This should be implemented by subclasses to define their forward pass.
        """
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for initialization."""
        config = self.wiring.get_config()
        config.update({
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
            'input_dim': self.input_size,
            'initializer': self.initializer,
        })
        return config

    def state_dict(self) -> Dict[str, Any]:
        """Return MLX parameters alongside high-level configuration."""
        config = {
            'units': self.units,
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
            'input_dim': self.input_size,
        }
        parameters = mxu.tree_map(lambda arr: mx.array(arr), self.parameters())
        return {'config': config, 'parameters': parameters}
        
    def _build_from_config(self, config: Dict[str, Any]) -> None:
        """Hook for subclasses to rebuild modules before loading weights."""
        return
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore configuration and MLX parameters from dictionary."""
        config = state_dict.get('config', {})
        if config:
            self.units = config['units']
            self.wiring = type(self.wiring).from_config(config['wiring'])
            self.activation_name = config['activation']
            self.activation = get_activation(self.activation_name)
            self.backbone_units = config['backbone_units']
            self.backbone_layers = config['backbone_layers']
            self.backbone_dropout = config['backbone_dropout']
            self.input_size = config.get('input_dim', self.input_size)
            self._build_from_config(config)
        parameters = state_dict.get('parameters', {})
        if parameters:
            self.update(parameters)


class LiquidRNN(nn.Module):
    """Base class for liquid neural networks."""
    
    def __init__(
        self,
        cell: LiquidCell,
        return_sequences: bool = False,
        return_state: bool = False,
        bidirectional: bool = False,
        merge_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode if bidirectional else None
        
        if bidirectional:
            # Create a new wiring instance with the same configuration
            wiring_config = cell.wiring.get_config()
            wiring_class = type(cell.wiring)
            
            # Build backward wiring with only the parameters the wiring class accepts
            backward_wiring_kwargs = {
                'units': wiring_config['units'],
            }
            
            # Add optional parameters if they exist in the config
            if 'output_dim' in wiring_config:
                backward_wiring_kwargs['output_dim'] = wiring_config['output_dim']
            if 'erev_init_seed' in wiring_config:
                backward_wiring_kwargs['erev_init_seed'] = wiring_config.get('random_seed', 1111)
            elif 'random_seed' in wiring_config:
                # Some wirings may use 'random_seed' instead of 'erev_init_seed'
                backward_wiring_kwargs['random_seed'] = wiring_config['random_seed']
            if 'self_connections' in wiring_config:
                backward_wiring_kwargs['self_connections'] = wiring_config.get('self_connections', True)
            
            backward_wiring = wiring_class(**backward_wiring_kwargs)
            
            if cell.wiring.input_dim is not None:
                backward_wiring.build(cell.wiring.input_dim)
            
            # Create the backward cell with the new wiring
            self.backward_cell = type(cell)(
                wiring=backward_wiring,
                activation=cell.activation_name,
                backbone_units=cell.backbone_units,
                backbone_layers=cell.backbone_layers,
                backbone_dropout=cell.backbone_dropout,
                initializer=cell.initializer
            )

    def __call__(
        self,
        inputs: mx.array,
        initial_states: Optional[List[mx.array]] = None,
        time_delta: Optional[Union[float, mx.array]] = None,
    ) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
        """Process sequence of inputs."""
        # Initialize states
        if initial_states is None:
            batch_size = inputs.shape[0]
            state = mx.zeros((batch_size, self.cell.units))
            if self.bidirectional:
                backward_state = mx.zeros((batch_size, self.cell.units))
        else:
            state = initial_states[0] if self.bidirectional else initial_states[0]
            if self.bidirectional:
                backward_state = initial_states[1]
        
        # Process sequence
        sequence_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        output_dim = self.cell.wiring.output_dim
        
        # Initialize outputs list
        outputs_list = []
            
        # Forward pass
        for t in range(sequence_length):
            current_input = inputs[:, t]
            current_time = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta or 1.0
            output, state = self.cell(current_input, state, time=current_time)
            outputs_list.append(output)
        
        # Backward pass if bidirectional
        if self.bidirectional:
            backward_outputs_list = []
            
            for t in range(sequence_length - 1, -1, -1):
                current_input = inputs[:, t]
                current_time = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta or 1.0
                output, backward_state = self.backward_cell(
                    current_input, 
                    backward_state, 
                    time=current_time
                )
                backward_outputs_list.append(output)
            
            # Reverse backward outputs to match forward sequence
            backward_outputs_list.reverse()
            
            # Stack and merge outputs
            if self.return_sequences:
                outputs = mx.stack(outputs_list, axis=1)
                backward_outputs = mx.stack(backward_outputs_list, axis=1)
            else:
                outputs = outputs_list[-1]
                backward_outputs = backward_outputs_list[-1]
                
            # Merge bidirectional outputs
            if self.merge_mode == "concat":
                outputs = mx.concatenate([outputs, backward_outputs], axis=-1)
            elif self.merge_mode == "sum":
                outputs = outputs + backward_outputs
            elif self.merge_mode == "mul":
                outputs = outputs * backward_outputs
            elif self.merge_mode == "ave":
                outputs = (outputs + backward_outputs) / 2
        else:
            # Stack outputs for sequence return or take last output
            if self.return_sequences:
                outputs = mx.stack(outputs_list, axis=1)
            else:
                outputs = outputs_list[-1]
        
        if self.return_state:
            if self.bidirectional:
                return outputs, [state, backward_state]
            return outputs, state
            
        return outputs

    def state_dict(self) -> Dict[str, Any]:
        """Serialize configuration and parameters."""
        config: Dict[str, Any] = {
            'cell': self.cell.state_dict(),
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'bidirectional': self.bidirectional,
        }
        if self.bidirectional:
            config['merge_mode'] = self.merge_mode
            config['backward_cell'] = self.backward_cell.state_dict()
        return {'config': config, 'parameters': {}}
        
    def process_time_delta(self, time_delta: Union[float, mx.array], batch_size: int, seq_len: int) -> mx.array:
        """Process time delta input into consistent format.
        
        Accepts:
        - Scalar (float/int): Broadcast to (batch_size, seq_len)
        - (seq_len,): Broadcast to (batch_size, seq_len)
        - (batch_size, seq_len): Use as-is
        - (batch_size, seq_len, 1): Squeeze last dimension
        """
        if isinstance(time_delta, (int, float)):
            return mx.full((batch_size, seq_len), time_delta)
        elif len(time_delta.shape) == 1:
            return mx.broadcast_to(time_delta[:, None], (batch_size, seq_len))
        elif len(time_delta.shape) == 2:
            return time_delta
        elif len(time_delta.shape) == 3 and time_delta.shape[2] == 1:
            # Handle (batch_size, seq_len, 1) by squeezing last dimension
            return mx.squeeze(time_delta, axis=2)
        else:
            raise ValueError(f"Invalid time_delta shape: {time_delta.shape}")
            
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore configuration and parameters."""
        config = state_dict.get('config', {})
        if config:
            self.cell.load_state_dict(config['cell'])
            self.return_sequences = config['return_sequences']
            self.return_state = config['return_state']
            self.bidirectional = config['bidirectional']
            if self.bidirectional and 'backward_cell' in config:
                self.merge_mode = config.get('merge_mode', self.merge_mode)
                self.backward_cell.load_state_dict(config['backward_cell'])
        parameters = state_dict.get('parameters', {})
        if parameters:
            self.update(parameters)
