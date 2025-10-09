"""Memory-Modulated RNN implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any

from .ltc_rnn import LTCRNN
from .ltc_rnn_cell import LTCRNNCell


class MMRNNCell(LTCRNNCell):
    """Memory-Modulated RNN Cell.
    
    This cell extends LTCRNNCell with memory modulation capabilities.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        bias: bool = True,
        activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        sparsity_mask: Optional[mx.array] = None,
    ):
        """Initialize the MMRNNCell.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            memory_size: Memory state dimension
            bias: Whether to use bias
            activation: Activation function name
            backbone_units: Units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone
            sparsity_mask: Optional sparsity mask
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            sparsity_mask=sparsity_mask,
        )
        self.memory_size = memory_size
        
        # Memory modulation parameters
        self.memory_gate = nn.Linear(hidden_size + input_size, memory_size)
        self.memory_update = nn.Linear(hidden_size + memory_size, hidden_size)
        
    def __call__(self, inputs, state=None, memory=None, time_delta=None):
        """Process one time step.
        
        Args:
            inputs: Input tensor
            state: Previous hidden state
            memory: Previous memory state
            time_delta: Optional time step size
            
        Returns:
            Tuple of (output, new_state, new_memory)
        """
        batch_size = inputs.shape[0]
        if state is None:
            state = mx.zeros((batch_size, self.hidden_size))
        if memory is None:
            memory = mx.zeros((batch_size, self.memory_size))
            
        # Regular RNN step
        output, new_state = super().__call__(inputs, state, time_delta)
        
        # Memory modulation
        combined = mx.concatenate([output, inputs], axis=-1)
        memory_gate = mx.sigmoid(self.memory_gate(combined))
        
        # Update memory
        new_memory = memory * memory_gate
        
        # Modulate hidden state with memory
        memory_combined = mx.concatenate([new_state, new_memory], axis=-1)
        final_state = new_state + mx.tanh(self.memory_update(memory_combined))
        
        return output, final_state, new_memory


class MMRNN(LTCRNN):
    """Memory-Modulated RNN layer.
    
    This layer extends LTCRNN with memory modulation capabilities.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        sparsity_mask: Optional[mx.array] = None,
    ):
        """Initialize MMRNN.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            memory_size: Memory state dimension
            num_layers: Number of stacked layers
            bias: Whether to use bias
            bidirectional: Process in both directions
            activation: Activation function name
            backbone_units: Units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone
            sparsity_mask: Optional sparsity mask
        """
        self.memory_size = memory_size
        
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            sparsity_mask=sparsity_mask,
        )
        
        # Replace LTCRNNCells with MMRNNCells
        self.forward_layers = []
        self.backward_layers = []
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            # Forward cell
            forward_cell = MMRNNCell(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                memory_size=memory_size,
                bias=bias,
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                sparsity_mask=sparsity_mask
            )
            setattr(self, f"forward_layer_{layer}", forward_cell)
            self.forward_layers.append(forward_cell)
            
            # Backward cell if bidirectional
            if bidirectional:
                backward_cell = MMRNNCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    memory_size=memory_size,
                    bias=bias,
                    activation=activation,
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_dropout=backbone_dropout,
                    sparsity_mask=sparsity_mask
                )
                setattr(self, f"backward_layer_{layer}", backward_cell)
                self.backward_layers.append(backward_cell)
    
    def __call__(
        self,
        x,
        initial_states=None,
        initial_memories=None,
        time_delta=None
    ):
        """Process input sequence.
        
        Args:
            x: Input tensor [batch, seq_len, features] or [seq_len, features]
            initial_states: Optional initial hidden states
            initial_memories: Optional initial memory states
            time_delta: Optional time steps between elements
            
        Returns:
            Tuple of:
            - Output tensor
            - List of final states
            - List of final memories
        """
        # Handle non-batched input
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len, _ = x.shape
        
        # Initialize states and memories if not provided
        if initial_states is None:
            initial_states = []
            for _ in range(self.num_layers):
                initial_states.append(mx.zeros((batch_size, self.hidden_size)))
                if self.bidirectional:
                    initial_states.append(mx.zeros((batch_size, self.hidden_size)))
                    
        if initial_memories is None:
            initial_memories = []
            for _ in range(self.num_layers):
                initial_memories.append(mx.zeros((batch_size, self.memory_size)))
                if self.bidirectional:
                    initial_memories.append(mx.zeros((batch_size, self.memory_size)))
        
        # Process each layer
        current_input = x
        final_states = []
        final_memories = []
        
        for layer in range(self.num_layers):
            forward_cell = self.forward_layers[layer]
            backward_cell = self.backward_layers[layer] if self.bidirectional else None
            
            # Forward pass
            forward_outputs = []
            state = initial_states[layer * (2 if self.bidirectional else 1)]
            memory = initial_memories[layer * (2 if self.bidirectional else 1)]
            
            for t in range(seq_len):
                dt = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta
                output, state, memory = forward_cell(
                    current_input[:, t],
                    state,
                    memory,
                    time_delta=dt
                )
                forward_outputs.append(output)
            
            forward_output = mx.stack(forward_outputs, axis=1)
            final_states.append(state)
            final_memories.append(memory)
            
            # Backward pass if bidirectional
            if self.bidirectional:
                backward_outputs = []
                state = initial_states[layer * 2 + 1]
                memory = initial_memories[layer * 2 + 1]
                
                for t in range(seq_len - 1, -1, -1):
                    dt = time_delta[:, t] if isinstance(time_delta, mx.array) else time_delta
                    output, state, memory = backward_cell(
                        current_input[:, t],
                        state,
                        memory,
                        time_delta=dt
                    )
                    backward_outputs.append(output)
                
                backward_output = mx.stack(backward_outputs[::-1], axis=1)
                final_states.append(state)
                final_memories.append(memory)
                
                # Combine forward and backward outputs
                current_input = mx.concatenate([forward_output, backward_output], axis=-1)
            else:
                current_input = forward_output
        
        # Handle non-batched output
        if squeeze_output:
            current_input = mx.squeeze(current_input, axis=0)
            final_states = [mx.squeeze(state, axis=0) for state in final_states]
            final_memories = [mx.squeeze(memory, axis=0) for memory in final_memories]
        
        return current_input, final_states, final_memories
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        state = super().state_dict()
        state['memory_size'] = self.memory_size
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load from state dictionary."""
        super().load_state_dict(state_dict)
        self.memory_size = state_dict['memory_size']
