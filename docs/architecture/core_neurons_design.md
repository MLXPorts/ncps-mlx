# Core Neurons Design: CfC and LTC

## Overview

Focus on implementing Keras-compatible CfC (Closed-form Continuous-time) and LTC (Linear Time-invariant Continuous-time) neurons as the foundation of NCPS. These neurons provide unique capabilities for continuous-time processing while maintaining familiar Keras-style interfaces.

## Core Components

### 1. CfC (Closed-form Continuous-time) Neuron
```python
class CfCCell(Layer):
    """Closed-form Continuous-time cell.
    
    Implements continuous-time processing with closed-form solution,
    providing efficient and stable temporal processing.
    
    Args:
        units: Number of units in the cell
        activation: Activation function ('tanh' default)
    """
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        
    def build(self, input_shape):
        # Standard Keras build pattern
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[-1], self.units]
        )
        self.recurrent_kernel = self.add_weight(
            'recurrent_kernel',
            shape=[self.units, self.units]
        )
```

### 2. LTC (Linear Time-invariant Continuous-time) Neuron
```python
class LTCCell(Layer):
    """Linear Time-invariant Continuous-time cell.
    
    Implements linear time-invariant dynamics for stable
    temporal processing.
    
    Args:
        units: Number of units in the cell
        timescale: Time constant for the cell
    """
    def __init__(self, units, timescale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.timescale = timescale
        
    def build(self, input_shape):
        # Standard Keras build pattern
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[-1], self.units]
        )
```

## Essential Features

### 1. Time-continuous Processing
```python
class TimeContiniousCell:
    """Mixin for time-continuous processing capabilities."""
    
    def process_time_step(self, inputs, states, dt):
        """Process a single time step.
        
        Args:
            inputs: Current input tensor
            states: Current state tensor
            dt: Time step size
        """
        pass
    
    def get_initial_state(self, inputs):
        """Get initial state for time-continuous processing."""
        pass
```

### 2. Basic Wiring Support
```python
class WirableCell:
    """Mixin for basic wiring capabilities."""
    
    def setup_wiring(self, input_size, hidden_size, output_size):
        """Setup basic wiring configuration."""
        pass
    
    def apply_wiring(self, connections):
        """Apply wiring pattern to the cell."""
        pass
```

## Implementation Strategy

### 1. Core Cell Implementation
```python
# Example CfC implementation
class CfCCell(Layer, TimeContiniousCell):
    def call(self, inputs, states):
        # Implement closed-form solution
        h_prev = states[0]
        z = K.dot(inputs, self.kernel)
        r = K.dot(h_prev, self.recurrent_kernel)
        
        # Closed-form update
        h = self.activation(z + r)
        return h, [h]
```

### 2. RNN Layer Wrapper
```python
class RNN(Layer):
    """RNN wrapper for time-continuous cells."""
    
    def __init__(self, cell, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
    
    def call(self, inputs):
        # Implement RNN processing
        states = self.cell.get_initial_state(inputs)
        outputs = []
        
        for t in range(inputs.shape[1]):
            output, states = self.cell(inputs[:, t], states)
            outputs.append(output)
            
        if self.return_sequences:
            return K.stack(outputs, axis=1)
        return outputs[-1]
```

## Usage Examples

### 1. Basic Usage
```python
# CfC example
model = Sequential([
    CfCCell(32),
    Dense(10)
])

# LTC example
model = Sequential([
    LTCCell(32, timescale=0.1),
    Dense(10)
])
```

### 2. Time Series Processing
```python
# Time series with CfC
model = Sequential([
    RNN(CfCCell(32), return_sequences=True),
    Dense(1)
])
```

## Core Optimizations

### 1. Temporal Processing
- Efficient time step handling
- Stable gradient computation
- Memory-efficient state management

### 2. Numerical Stability
- Careful initialization
- Gradient clipping
- State normalization

## Testing Focus

### 1. Core Functionality
```python
def test_cfc_cell():
    cell = CfCCell(32)
    inputs = np.random.random((32, 10))
    output = cell(inputs)
    assert output.shape == (32, 32)
```

### 2. Temporal Stability
```python
def test_temporal_stability():
    cell = LTCCell(32)
    sequence = np.random.random((32, 100, 10))
    states = []
    
    for t in range(100):
        state = cell(sequence[:, t])
        states.append(state)
    
    # Check stability
    assert is_stable(states)
```

## Next Steps

1. Implement core CfC cell
2. Implement core LTC cell
3. Add basic wiring support
4. Create RNN wrapper
5. Add essential optimizations
6. Write core tests

## Future Extensions

1. Advanced wiring patterns
2. Additional cell types
3. Performance optimizations
4. Enhanced stability features

This focused approach allows us to:
- Build on proven CfC and LTC foundations
- Maintain Keras compatibility
- Provide essential functionality
- Enable future extensions