# LTC Implementation Plan

## Overview

Implement LTC (Linear Time-invariant Continuous-time) cell on top of BaseCell, matching the production implementation while using Keras 3.8 patterns.

## Core Implementation

### 1. Class Definition
```python
# ncps/layers/ltc.py

import keras
from .base import BaseCell

@keras.saving.register_keras_serializable(package="ncps")
class LTCCell(BaseCell):
    """Linear Time-invariant Continuous-time cell."""
    
    def __init__(
        self,
        units,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs
    ):
        super().__init__(units, **kwargs)
        
        self.activation = keras.activations.get(activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone_fn = None
        
        # Calculate backbone dimensions
        if backbone_layers > 0 and backbone_units:
            self.backbone_output_dim = backbone_units
        else:
            self.backbone_output_dim = None
```

### 2. Build Method
```python
def build(self, input_shape):
    """Initialize weights and layers."""
    super().build(input_shape)
    
    # Get input dimension
    input_dim = input_shape[-1]
    
    # Build backbone if needed
    if self.backbone_layers > 0:
        backbone_layers = []
        for i in range(self.backbone_layers):
            backbone_layers.append(
                keras.layers.Dense(
                    self.backbone_units,
                    self.activation,
                    name=f"backbone{i}"
                )
            )
            if self.backbone_dropout > 0:
                backbone_layers.append(
                    keras.layers.Dropout(self.backbone_dropout)
                )
        
        self.backbone_fn = keras.Sequential(backbone_layers)
        self.backbone_fn.build((None, self.units + input_dim))
        cat_shape = self.backbone_units
    else:
        cat_shape = self.units + input_dim
    
    # Initialize main weights
    self.kernel = self.add_weight(
        shape=(cat_shape, self.units),
        initializer="glorot_uniform",
        name="kernel"
    )
    self.bias = self.add_weight(
        shape=(self.units,),
        initializer="zeros",
        name="bias"
    )
    
    # Initialize time constant network
    self.tau_kernel = keras.layers.Dense(
        self.units,
        name="tau_kernel"
    )
    self.tau_kernel.build((None, cat_shape))
```

### 3. Call Method
```python
def call(self, inputs, states, training=None):
    """Process one timestep."""
    # Get current state
    state = states[0]
    
    # Handle time input
    if isinstance(inputs, (tuple, list)):
        inputs, t = inputs
        t = keras.ops.reshape(t, [-1, 1])
    else:
        t = 1.0
    
    # Combine inputs and state
    x = keras.layers.Concatenate()([inputs, state])
    
    # Apply backbone if present
    if self.backbone_fn is not None:
        x = self.backbone_fn(x, training=training)
    
    # Compute delta term
    d = keras.ops.matmul(x, self.kernel) + self.bias
    
    # Compute time constants
    tau = keras.ops.exp(self.tau_kernel(x))
    
    # Update state using time constant
    new_state = state + t * (-state + d) / tau
    
    # Apply activation
    output = self.activation(new_state)
    
    return output, [new_state]
```

### 4. Configuration Methods
```python
def get_config(self):
    """Get layer configuration."""
    config = super().get_config()
    config.update({
        'activation': keras.activations.serialize(self.activation),
        'backbone_units': self.backbone_units,
        'backbone_layers': self.backbone_layers,
        'backbone_dropout': self.backbone_dropout
    })
    return config

@classmethod
def from_config(cls, config):
    """Create layer from configuration."""
    return cls(**config)
```

## Testing Strategy

### 1. Basic Tests
```python
def test_ltc_basics():
    """Test basic LTC functionality."""
    cell = LTCCell(32)
    output, state = cell(inputs, [initial_state])
    assert output.shape == (batch_size, 32)
    assert state[0].shape == (batch_size, 32)
```

### 2. Time Constant Tests
```python
def test_time_constants():
    """Test time constant behavior."""
    cell = LTCCell(32)
    
    # Test with different time steps
    output1, state1 = cell([inputs, 0.1], [initial_state])
    output2, state2 = cell([inputs, 1.0], [initial_state])
    
    # States should be different due to time scaling
    assert not keras.ops.allclose(state1[0], state2[0])
```

### 3. Backbone Tests
```python
def test_backbone():
    """Test backbone network."""
    cell = LTCCell(
        32,
        backbone_units=64,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    
    # Test with and without training
    out1, _ = cell(inputs, [initial_state], training=True)
    out2, _ = cell(inputs, [initial_state], training=False)
    
    # Outputs should differ due to dropout
    assert not keras.ops.allclose(out1, out2)
```

## Success Criteria

### 1. Functionality
- Time constant processing works
- State updates correctly
- Backbone network functions
- Activation applies properly

### 2. Compatibility
- Works with Keras 3.8 RNN
- Matches production behavior
- Supports all features

### 3. Code Quality
- Clean implementation
- Good test coverage
- Clear documentation

## Next Steps

1. Implement core class
2. Add build method
3. Add call method
4. Implement time constant handling
5. Add tests
6. Document thoroughly

## Usage Examples

### 1. Basic Usage
```python
# Create LTC model
model = keras.Sequential([
    keras.layers.rnn.RNN(LTCCell(32)),
    keras.layers.Dense(10)
])
```

### 2. With Backbone
```python
# LTC with processing backbone
cell = LTCCell(
    32,
    backbone_units=64,
    backbone_layers=2,
    backbone_dropout=0.1
)

# Use in model
model = keras.Sequential([
    keras.layers.rnn.RNN(cell),
    keras.layers.Dense(10)
])
```

### 3. With Time Steps
```python
# Create cell
cell = LTCCell(32)

# Process with explicit time steps
output, state = cell(
    [inputs, time_steps],
    [initial_state]
)