"""Base classes for neural circuit implementations in Keras."""

import keras
from keras import ops, layers
from typing import Optional, List, Dict, Any, Union, Tuple


class LiquidCell(layers.Layer):
    """Base class for liquid time-constant cells."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = None
        self.output_size = None
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Create initial state tensor.
        
        Args:
            inputs: Input tensor if available
            batch_size: Batch size if inputs not available
            dtype: Data type to use
            
        Returns:
            Initial state tensor
        """
        if batch_size is None:
            batch_size = keras.ops.shape(inputs)[0]
        if dtype is None:
            dtype = keras.backend.floatx()
            
        return [
            keras.ops.zeros((batch_size, dim), dtype=dtype)
            for dim in (self.state_size if isinstance(self.state_size, (list, tuple))
                       else [self.state_size])
        ]


class LiquidRNN(layers.RNN):
    """Base class for liquid time-constant RNNs."""
    
    def __init__(
        self,
        cell,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs
    ):
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )
    
    def build(self, input_shape):
        """Build the RNN layer.
        
        Args:
            input_shape: Input shape tuple
        """
        # Ensure input_shape is a tuple
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
            
        batch_size = input_shape[0] if input_shape[0] else None
        self.input_spec = keras.layers.InputSpec(shape=(batch_size, None, input_shape[-1]))
        
        step_input_shape = (input_shape[0],) + input_shape[2:]
        self.cell.build(step_input_shape)
        
        self.built = True
    
    def call(self, inputs, mask=None, training=None, initial_state=None):
        """Process input sequence.
        
        Args:
            inputs: Input tensor
            mask: Optional mask tensor
            training: Whether in training mode
            initial_state: Optional initial state
            
        Returns:
            Output tensor or tuple of output and states
        """
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state
        )