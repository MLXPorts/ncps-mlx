"""Wired CfC cell implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Union

from .cfc_cell_mlx import CfCCell


class WiredCfCCell(nn.Module):
    """
    Wired Closed-form Continuous-time (CfC) cell.
    
    This class implements a CfC cell with wiring constraints, allowing for
    structured connectivity between neurons.
    """
    
    def __init__(
        self,
        wiring,
        fully_recurrent: bool = True,
        mode: str = "default",
        activation: str = "lecun_tanh",
        **kwargs
    ):
        """
        Initialize WiredCfCCell.
        
        Args:
            wiring: Neural wiring pattern
            fully_recurrent: Whether to use full recurrent connectivity
            mode: Operation mode ("default", "pure", "no_gate")
            activation: Activation function name
            **kwargs: Additional arguments
        """
        super().__init__()
        self.wiring = wiring
        
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
            
        self.mode = mode
        self.fully_recurrent = fully_recurrent
        self.activation = activation
        self._cfc_layers: List[CfCCell] = []
        self._layer_sizes: List[int] = []
        
        # Build the wiring if input dimension is known
        if self.wiring.input_dim is not None:
            self.build()

    @property
    def state_size(self) -> int:
        """Get state size."""
        return self.wiring.units

    @property
    def input_size(self) -> int:
        """Get input size."""
        return self.wiring.input_dim

    @property
    def output_size(self) -> int:
        """Get output size."""
        return self.wiring.output_dim

    def build(self):
        """Build the wired CfC cell."""
        for i in range(self.wiring.num_layers):
            layer_i_neurons = self.wiring.get_neurons_of_layer(i)
            
            # Get input sparsity mask
            if i == 0:
                input_sparsity = self.wiring.sensory_adjacency_matrix[:, layer_i_neurons]
            else:
                prev_layer_neurons = self.wiring.get_neurons_of_layer(i - 1)
                input_sparsity = self.wiring.adjacency_matrix[:, layer_i_neurons]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            
            # Get recurrent sparsity mask
            if self.fully_recurrent:
                recurrent_sparsity = mx.ones((len(layer_i_neurons), len(layer_i_neurons)))
            else:
                recurrent_sparsity = self.wiring.adjacency_matrix[layer_i_neurons, layer_i_neurons]
            
            # Combine masks
            sparsity_mask = mx.concatenate([input_sparsity, recurrent_sparsity], axis=0)
            
            # Create CfC cell
            cell = CfCCell(
                len(layer_i_neurons),
                mode=self.mode,
                activation=self.activation,
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0,
                sparsity_mask=sparsity_mask,
            )
            
            self._cfc_layers.append(cell)
            
        self._layer_sizes = [l.units for l in self._cfc_layers]

    def __call__(
        self,
        inputs: Union[mx.array, Tuple[mx.array, mx.array]],
        states: List[mx.array],
        **kwargs
    ) -> Tuple[mx.array, mx.array]:
        """
        Process one time step.
        
        Args:
            inputs: Input tensor or tuple of (inputs, time)
            states: List of state tensors
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, new_states)
        """
        # Handle time input
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = mx.reshape(t, [-1, 1])
        else:
            t = 1.0
            
        # Split states for each layer
        states = mx.split(states[0], self._layer_sizes, axis=-1)
        assert len(states) == self.wiring.num_layers, \
            f'Incompatible num of states [{len(states)}] and wiring layers [{self.wiring.num_layers}]'
            
        # Process each layer
        new_hiddens = []
        for i, cfc_layer in enumerate(self._cfc_layers):
            if t == 1.0:
                output, new_hidden = cfc_layer(inputs, states[i], time=t)
            else:
                output, new_hidden = cfc_layer(inputs, states[i], time=t)
            new_hiddens.append(new_hidden)
            inputs = output
            
        assert len(new_hiddens) == self.wiring.num_layers, \
            f'Internal error new_hiddens [{len(new_hiddens)}] != num_layers [{self.wiring.num_layers}]'
            
        # Handle output dimension
        if self.wiring.output_dim != output.shape[-1]:
            output = output[:, :self.wiring.output_dim]
            
        # Combine hidden states
        new_hiddens = mx.concatenate(new_hiddens, axis=-1)
        return output, new_hiddens

    def state_dict(self) -> dict:
        """Get state dictionary."""
        return {
            "wiring": self.wiring,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
            "activation": self.activation,
            "cfc_layers": [layer.state_dict() for layer in self._cfc_layers],
            "layer_sizes": self._layer_sizes,
        }

    def load_state_dict(self, state_dict: dict):
        """Load from state dictionary."""
        self.wiring = state_dict["wiring"]
        self.fully_recurrent = state_dict["fully_recurrent"]
        self.mode = state_dict["mode"]
        self.activation = state_dict["activation"]
        self._layer_sizes = state_dict["layer_sizes"]
        
        for i, layer_state in enumerate(state_dict["cfc_layers"]):
            self._cfc_layers[i].load_state_dict(layer_state)
