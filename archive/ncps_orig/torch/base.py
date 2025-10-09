"""Base classes for PyTorch neural circuit implementations."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Tuple

class LiquidCell(nn.Module):
    """Base class for liquid neural network cells.
    
    Provides minimal shared functionality for dimension tracking,
    configuration management, and basic initialization patterns.
    
    Args:
        wiring: Neural circuit wiring specification
        activation: Activation function name or callable
        backbone_units: List of backbone layer sizes
        backbone_layers: Number of backbone layers
        backbone_dropout: Backbone dropout rate
    """
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
    ):
        super().__init__()
        self.wiring = wiring
        self.units = wiring.units
        self.hidden_size = wiring.units
        self.input_size = wiring.input_dim if wiring.input_dim is not None else 0
        
        # Get activation function
        self.activation_name = activation
        if isinstance(activation, str):
            if activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "sigmoid":
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
            
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
        
        # Build layers
        for i, units in enumerate(self.backbone_units):
            # Add linear layer
            layers.append(nn.Linear(current_dim, units))
            
            # Add activation and dropout except for last layer
            if i < len(self.backbone_units) - 1:
                layers.append(self.activation)
                if self.backbone_dropout > 0:
                    layers.append(nn.Dropout(p=self.backbone_dropout))
            
            current_dim = units
            
        self.backbone = nn.Sequential(*layers) if layers else None

    def apply_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backbone network if present.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        if self.backbone is not None:
            # Get input shape
            batch_size = x.shape[0]
            
            # Ensure input has correct shape
            if x.shape[-1] != self.backbone_input_dim:
                raise ValueError(
                    f"Expected input dimension {self.backbone_input_dim}, "
                    f"got {x.shape[-1]}"
                )
            
            # Apply backbone
            x = self.backbone(x)
            
        return x

    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor,
        time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one step with the cell.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            state: Previous state tensor of shape [batch_size, hidden_size]
            time: Time delta since last update
            
        Returns:
            Tuple of (output, new_state) tensors
        """
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get serializable state."""
        state = {
            'units': self.units,
            'wiring': self.wiring.get_config(),
            'activation': self.activation_name,
            'backbone_units': self.backbone_units,
            'backbone_layers': self.backbone_layers,
            'backbone_dropout': self.backbone_dropout,
        }
        if self.backbone is not None:
            state['backbone'] = self.backbone.state_dict()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load cell state from dictionary."""
        self.units = state_dict['units']
        self.wiring = type(self.wiring).from_config(state_dict['wiring'])
        self.activation_name = state_dict['activation']
        if isinstance(self.activation_name, str):
            if self.activation_name == "tanh":
                self.activation = nn.Tanh()
            elif self.activation_name == "relu":
                self.activation = nn.ReLU()
            elif self.activation_name == "sigmoid":
                self.activation = nn.Sigmoid()
        self.backbone_units = state_dict['backbone_units']
        self.backbone_layers = state_dict['backbone_layers']
        self.backbone_dropout = state_dict['backbone_dropout']
        
        if 'backbone' in state_dict:
            self.build_backbone()
            self.backbone.load_state_dict(state_dict['backbone'])


class LiquidRNN(nn.Module):
    """Base class for liquid neural networks.
    
    Wraps a LiquidCell to process sequences of inputs.
    
    Args:
        cell: The liquid cell to use
        return_sequences: Whether to return full sequence or just final output
        return_state: Whether to return final state
        bidirectional: Whether to process sequence bidirectionally
        merge_mode: How to merge bidirectional outputs (concat, sum, mul, ave)
    """
    
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
            # Create backward cell with same config
            wiring_config = cell.wiring.get_config()
            wiring_class = type(cell.wiring)
            backward_wiring = wiring_class.from_config(wiring_config)
            
            self.backward_cell = type(cell)(
                wiring=backward_wiring,
                activation=cell.activation_name,
                backbone_units=cell.backbone_units,
                backbone_layers=cell.backbone_layers,
                backbone_dropout=cell.backbone_dropout
            )

    def forward(
        self,
        inputs: torch.Tensor,
        initial_states: Optional[List[torch.Tensor]] = None,
        time_delta: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Process sequence of inputs.
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_size]
            initial_states: Optional list of initial state tensors
            time_delta: Optional time steps between inputs
            
        Returns:
            Output tensor or tuple of (output tensor, final states)
        """
        # Initialize states
        if initial_states is None:
            batch_size = inputs.size(0)
            state = torch.zeros(
                batch_size,
                self.cell.units,
                device=inputs.device,
                dtype=inputs.dtype
            )
            if self.bidirectional:
                backward_state = torch.zeros(
                    batch_size,
                    self.cell.units,
                    device=inputs.device,
                    dtype=inputs.dtype
                )
        else:
            state = initial_states[0]
            if self.bidirectional:
                backward_state = initial_states[1]
        
        # Process sequence
        outputs_list = []
        for t in range(inputs.size(1)):
            current_input = inputs[:, t]
            current_time = (
                time_delta[:, t] if isinstance(time_delta, torch.Tensor)
                else time_delta or 1.0
            )
            output, state = self.cell(current_input, state, time=current_time)
            outputs_list.append(output)
            
        # Process backward direction if needed
        if self.bidirectional:
            backward_outputs_list = []
            for t in range(inputs.size(1) - 1, -1, -1):
                current_input = inputs[:, t]
                current_time = (
                    time_delta[:, t] if isinstance(time_delta, torch.Tensor)
                    else time_delta or 1.0
                )
                output, backward_state = self.backward_cell(
                    current_input,
                    backward_state,
                    time=current_time
                )
                backward_outputs_list.append(output)
                
            # Reverse backward outputs
            backward_outputs_list.reverse()
            
            # Stack and merge outputs
            if self.return_sequences:
                outputs = torch.stack(outputs_list, dim=1)
                backward_outputs = torch.stack(backward_outputs_list, dim=1)
            else:
                outputs = outputs_list[-1]
                backward_outputs = backward_outputs_list[-1]
                
            # Merge bidirectional outputs
            if self.merge_mode == "concat":
                outputs = torch.cat([outputs, backward_outputs], dim=-1)
            elif self.merge_mode == "sum":
                outputs = outputs + backward_outputs
            elif self.merge_mode == "mul":
                outputs = outputs * backward_outputs
            elif self.merge_mode == "ave":
                outputs = (outputs + backward_outputs) / 2
        else:
            # Stack outputs for sequence return or take last output
            if self.return_sequences:
                outputs = torch.stack(outputs_list, dim=1)
            else:
                outputs = outputs_list[-1]
        
        if self.return_state:
            if self.bidirectional:
                return outputs, [state, backward_state]
            return outputs, state
            
        return outputs

    def state_dict(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        state = {
            'cell': self.cell.state_dict(),
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'bidirectional': self.bidirectional,
        }
        if self.bidirectional:
            state['merge_mode'] = self.merge_mode
            state['backward_cell'] = self.backward_cell.state_dict()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.cell.load_state_dict(state_dict['cell'])
        self.return_sequences = state_dict['return_sequences']
        self.return_state = state_dict['return_state']
        self.bidirectional = state_dict['bidirectional']
        
        if self.bidirectional:
            self.merge_mode = state_dict['merge_mode']
            self.backward_cell.load_state_dict(state_dict['backward_cell'])