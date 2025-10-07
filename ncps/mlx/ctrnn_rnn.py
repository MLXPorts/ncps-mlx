"""Continuous-Time Recurrent Neural Network (CTRNN) RNN implementation."""

import mlx.core as mx
from typing import Optional, List, Union, Callable, Tuple

from .base import LiquidRNN
from .liquid_utils import TimeAwareMixin
from .ctrnn import CTRNNCell


class CTRNN(LiquidRNN):
    """A Continuous-Time Recurrent Neural Network (CTRNN) layer."""
    
    def __init__(
        self,
        units: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        return_sequences: bool = True,
        return_state: bool = False,
        global_feedback: bool = False,
        activation: Callable = mx.tanh,
        cell_clip: Optional[float] = None,
    ):
        """Initialize the CTRNN layer.
        
        Args:
            units: Number of units in each layer
            num_layers: Number of stacked CTRNN layers
            bidirectional: Whether to use bidirectional processing
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final states
            global_feedback: Whether to use global feedback
            activation: Activation function to use
            cell_clip: Optional value to clip cell outputs
        """
        # Create base cell
        cell = CTRNNCell(
            units=units,
            global_feedback=global_feedback,
            activation=activation,
            cell_clip=cell_clip,
        )

        self.num_layers = num_layers
        self.hidden_size = units
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            bidirectional=bidirectional,
            merge_mode="concat" if bidirectional else None,
        )
        
        # Create forward layers
        forward_layers = []
        for i in range(num_layers):
            layer_cell = CTRNNCell(
                units=units,
                global_feedback=global_feedback,
                activation=activation,
                cell_clip=cell_clip,
            )
            # Register layer as a submodule
            setattr(self, f"forward_layer_{i}", layer_cell)
            forward_layers.append(layer_cell)
        self.forward_layers = forward_layers
        
        # Create backward layers if bidirectional
        if bidirectional:
            backward_layers = []
            for i in range(num_layers):
                layer_cell = CTRNNCell(
                    units=units,
                    global_feedback=global_feedback,
                    activation=activation,
                    cell_clip=cell_clip,
                )
                # Register layer as a submodule
                setattr(self, f"backward_layer_{i}", layer_cell)
                backward_layers.append(layer_cell)
            self.backward_layers = backward_layers
    
    def __call__(
        self,
        x: mx.array,
        initial_states: Optional[List[mx.array]] = None,
        time_delta: Optional[Union[float, mx.array]] = None,
    ) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
        """Process a sequence through the CTRNN network.
        
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
                output, [state] = forward_cell(current_input[:, t], state, time=dt)
                forward_states.append(output)
            
            forward_output = mx.stack(forward_states, axis=1)
            final_states.append(state)
            
            # Backward pass if bidirectional
            if self.bidirectional:
                backward_states = []
                state = initial_states[layer * 2 + 1]
                
                for t in range(seq_len - 1, -1, -1):
                    dt = time_delta[:, t] if time_delta is not None else 1.0
                    output, [state] = backward_cell(current_input[:, t], state, time=dt)
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