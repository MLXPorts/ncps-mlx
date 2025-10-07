"""Basic RNN layer implementation."""

import numpy as np
from ncps import ops
from typing import Optional, List, Union, Tuple, Any, Dict, Type

from .layer import Layer


class RNNCell(Layer):
    """Base class for RNN cells."""
    
    def __init__(self, units: int, dtype: str = "float32", **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.state_size = units
    
    def get_initial_state(self, batch_size: Optional[int] = None) -> List[Any]:
        """Get initial state for RNN."""
        return [ops.zeros((batch_size, self.state_size), dtype=self.dtype)]
    
    def call(self, inputs, states, **kwargs) -> Tuple[Any, List[Any]]:
        """Process one timestep."""
        raise NotImplementedError()
    
    def get_config(self):
        """Get cell configuration."""
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config


# Registry for cell types
CELL_TYPES: Dict[str, Type[RNNCell]] = {}


def register_cell(cls: Type[RNNCell]):
    """Register a cell type."""
    CELL_TYPES[cls.__name__] = cls
    return cls


class RNN(Layer):
    """RNN layer that processes sequences using a cell."""
    
    def __init__(
        self,
        cell: RNNCell,
        return_sequences: bool = False,
        return_state: bool = False,
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # For tracking state
        self.states = None
        self.batch_size = None
        self.timesteps = None
        
        # Register cell type if needed
        cell_type = type(cell)
        if cell_type.__name__ not in CELL_TYPES:
            register_cell(cell_type)
    
    def _get_shape(self, x):
        """Helper to get shape from tensor or shape tuple."""
        if hasattr(x, 'shape'):
            return x.shape
        if isinstance(x, (list, tuple)):
            return tuple(x)
        if isinstance(x, int):
            return (x,)
        raise ValueError(f"Cannot get shape from {type(x)}: {x}")
    
    def build(self, input_shape):
        """Build the layer."""
        # Handle list inputs
        if isinstance(input_shape, (list, tuple)) and not isinstance(input_shape[0], int):
            # Get shapes from inputs
            x_shape = self._get_shape(input_shape[0])
            t_shape = self._get_shape(input_shape[1])
            
            # Ensure shapes are compatible
            if x_shape[0] != t_shape[0] or x_shape[1] != t_shape[1]:
                raise ValueError(
                    f"Input and time shapes must match in batch and time dimensions. "
                    f"Got input shape {x_shape} and time shape {t_shape}"
                )
            input_shape = x_shape
        else:
            input_shape = self._get_shape(input_shape)
        
        # Ensure input is 3D (batch, time, features)
        if len(input_shape) != 3:
            raise ValueError(
                f"Input should be 3D (batch, time, features), got shape: {input_shape}"
            )
            
        # Build the cell with single timestep shape
        self.cell.build((input_shape[0], input_shape[2]))
        self.built = True
    
    def call(
        self,
        inputs,
        initial_state: Optional[List[Any]] = None,
        training: Optional[bool] = None,
        **kwargs
    ):
        """Process input sequence."""
        # Handle list inputs
        if isinstance(inputs, (list, tuple)):
            x, t = inputs
            # Ensure time has correct shape
            if len(t.shape) == 2:
                t = ops.expand_dims(t, -1)
        else:
            x, t = inputs, None
            
        # Get dimensions
        self.batch_size = x.shape[0]
        self.timesteps = x.shape[1]
        
        # Get initial state
        if initial_state is None:
            self.states = self.cell.get_initial_state(self.batch_size)
        else:
            self.states = initial_state
            
        # Container for outputs
        outputs = []
        
        # Process each timestep
        for i in range(self.timesteps):
            # Get current input and time
            current_input = x[:, i]
            current_time = t[:, i] if t is not None else None
            
            # Create cell inputs
            if current_time is not None:
                cell_inputs = [current_input, current_time]
            else:
                cell_inputs = current_input
                
            # Process step
            output, self.states = self.cell(
                cell_inputs,
                self.states,
                training=training,
                **kwargs
            )
            outputs.append(output)
            
        # Stack outputs if returning sequences
        if self.return_sequences:
            outputs = ops.stack(outputs, axis=1)
        else:
            outputs = outputs[-1]
            
        if self.return_state:
            return outputs, self.states
        return outputs
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'cell': {
                'class_name': self.cell.__class__.__name__,
                'config': self.cell.get_config()
            },
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        # Extract cell config
        cell_config = config.pop('cell')
        cell_cls = CELL_TYPES[cell_config['class_name']]
        cell = cell_cls.from_config(cell_config['config'])
        
        # Create layer
        return cls(cell=cell, **config)