"""Liquid Time-Constant (LTC) RNN implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any

from .base import LiquidRNN
from .liquid_utils import TimeAwareMixin
from .ltc_cell import LTCCell


class LTC(LiquidRNN):
    """A Liquid Time-Constant (LTC) RNN layer."""
    
    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        wiring = None,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        return_sequences: bool = True,  # Changed default to True to match test expectations
        return_state: bool = False,
        activation: str = "identity",
        backbone_units: Optional[Union[int, List[int]]] = None,
        backbone_layers: int = 0,  # Changed default to 0 to avoid dimension issues when not needed
        backbone_dropout: float = 0.0,
        sparsity_mask: Optional[mx.array] = None,
        ode_solver: str = "semi_implicit",
        ode_solver_unfolds: int = 6,
        input_mapping: str = "affine",
        **ltc_cell_kwargs,
    ):
        """Initialize the LTC layer."""
        if wiring is not None:
            if input_size is not None or hidden_size is not None:
                raise ValueError("Cannot specify both wiring and input_size/hidden_size")
            if wiring.input_dim is None:
                if input_size is None:
                    raise ValueError("Must specify input_size when wiring.input_dim is None")
                wiring.build(input_size)
            input_size = wiring.input_dim
            hidden_size = wiring.units
            sparsity_mask = wiring.adjacency_matrix
        elif input_size is None or hidden_size is None:
            raise ValueError("Must specify either wiring or both input_size and hidden_size")

        # Create wiring if not provided
        if wiring is None:
            from .wirings import FullyConnected
            wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
            wiring.build(input_size)

        # Process backbone units
        if backbone_units is None:
            backbone_units = []
        elif isinstance(backbone_units, int):
            backbone_units = [backbone_units]

        # Create LTC cell
        cell = LTCCell(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            ode_solver=ode_solver,
            ode_solver_unfolds=ode_solver_unfolds,
            input_mapping=input_mapping,
            **ltc_cell_kwargs,
        )

        self.num_layers = num_layers
        self.hidden_size = wiring.units
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            bidirectional=bidirectional,
            merge_mode="concat" if bidirectional else None,
        )
        
        # Create forward layers
        self.forward_layers = []
        for idx in range(num_layers):
            layer_cell = LTCCell(
                wiring=type(wiring).from_config(wiring.get_config()),
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                ode_solver=ode_solver,
                ode_solver_unfolds=ode_solver_unfolds,
                input_mapping=input_mapping,
                **ltc_cell_kwargs,
            )
            setattr(self, f"forward_layer_{idx}", layer_cell)
            self.forward_layers.append(layer_cell)
        
        # Create backward layers if bidirectional
        if bidirectional:
            self.backward_layers = []
            for idx in range(num_layers):
                layer_cell = LTCCell(
                    wiring=type(wiring).from_config(wiring.get_config()),
                    activation=activation,
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_dropout=backbone_dropout,
                    ode_solver=ode_solver,
                    ode_solver_unfolds=ode_solver_unfolds,
                    input_mapping=input_mapping,
                    **ltc_cell_kwargs,
                )
                setattr(self, f"backward_layer_{idx}", layer_cell)
                self.backward_layers.append(layer_cell)
    
    def __call__(
        self,
        x: mx.array,
        initial_states: Optional[List[mx.array]] = None,
        time_delta: Optional[Union[float, mx.array]] = None,
    ) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
        """Process a sequence through the LTC network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            initial_states: Optional list of initial states for each layer
            time_delta: Optional time steps between sequence elements
            
        Returns:
            If return_sequences is True, returns sequences of shape [batch_size, seq_len, hidden_size],
            otherwise returns the last output of shape [batch_size, hidden_size].
            If return_state is True, also returns the final states.
        """
        batch_size, seq_len, _ = x.shape
        
        # Process time delta
        if time_delta is not None:
            time_delta = self.process_time_delta(time_delta, batch_size, seq_len)
        
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
                dt = time_delta[:, t] if time_delta is not None else 1.0
                output, state = forward_cell(current_input[:, t], state, time=dt)
                forward_states.append(output)
            
            forward_output = mx.stack(forward_states, axis=1)
            final_states.append(state)
            
            # Backward pass if bidirectional
            if self.bidirectional:
                backward_states = []
                state = initial_states[layer * 2 + 1]
                
                for t in range(seq_len - 1, -1, -1):
                    dt = time_delta[:, t] if time_delta is not None else 1.0
                    output, state = backward_cell(current_input[:, t], state, time=dt)
                    backward_states.append(output)
                
                backward_output = mx.stack(backward_states[::-1], axis=1)
                final_states.append(state)
                
                # Combine forward and backward outputs
                current_input = mx.concatenate([forward_output, backward_output], axis=-1)
            else:
                current_input = forward_output
        
        # Prepare output
        if not self.return_sequences:
            current_input = current_input[:, -1]
            
        if self.return_state:
            return current_input, final_states
        return current_input
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        config = state['config']
        config.update({
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'forward_layers': [layer.state_dict() for layer in self.forward_layers],
        })
        if self.bidirectional:
            config['backward_layers'] = [layer.state_dict() for layer in self.backward_layers]
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        config = state_dict.get('config', {})
        if config:
            self.num_layers = config.get('num_layers', self.num_layers)
            self.hidden_size = config.get('hidden_size', self.hidden_size)
            forward_states = config.get('forward_layers', [])
            for layer, layer_state in zip(self.forward_layers, forward_states):
                layer.load_state_dict(layer_state)
            if self.bidirectional:
                backward_states = config.get('backward_layers', [])
                for layer, layer_state in zip(self.backward_layers, backward_states):
                    layer.load_state_dict(layer_state)

    # ------------------------------------------------------------------ #
    # LTC-specific utilities
    # ------------------------------------------------------------------ #
    def set_solver(self, solver: str) -> None:
        self.cell.set_solver(solver)
        for layer in self.forward_layers:
            layer.set_solver(solver)
        if self.bidirectional:
            for layer in self.backward_layers:
                layer.set_solver(solver)

    def apply_constraints(self) -> None:
        self.cell.apply_constraints()
        for layer in self.forward_layers:
            layer.apply_constraints()
        if self.bidirectional:
            for layer in self.backward_layers:
                layer.apply_constraints()
