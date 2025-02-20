# MLX Layer Design for Neural Circuit Policies

## Overview

Design for a layer system that wraps MLX functionality while providing Keras-like usability, specifically focused on Neural Circuit Policy requirements.

## Core Design

### 1. MLX Integration

```python
class NCPLayer(nn.Module):
    """Base layer class wrapping MLX functionality.
    
    Provides Keras-like interface while maintaining MLX's performance
    characteristics and native integration.
    """
    
    def __init__(self):
        super().__init__()
        self.built = False
        self._input_shape = None
        
    def build(self, input_shape):
        """Lazy initialization following Keras pattern."""
        self._input_shape = input_shape
        self.built = True
        
    def __call__(self, x):
        """MLX-native forward pass."""
        if not self.built:
            self.build(x.shape)
        return super().__call__(x)
```

### 2. Layer Types

```python
class NCPDense(NCPLayer):
    """Dense layer with NCP-specific functionality.
    
    Wraps MLX's Linear layer while adding:
    - Wiring pattern support
    - Shape inference
    - State management
    """
    
    def __init__(self, units, activation=None, use_bias=True):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        
    def build(self, input_shape):
        self.layer = nn.Linear(input_shape[-1], self.units, bias=self.use_bias)
        super().build(input_shape)
```

### 3. RNN Support

```python
class NCPRNNCell(NCPLayer):
    """Base RNN cell with NCP extensions.
    
    Provides:
    - MLX-native operations
    - State management
    - Wiring integration
    """
    
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.state_size = units
        
    def __call__(self, inputs, states):
        """MLX-native step function."""
        raise NotImplementedError()
```

## Key Features

### 1. MLX Integration
- Native MLX operations
- Direct access to MLX's optimizations
- Proper gradient handling

### 2. Shape Management
```python
def compute_output_shape(self, input_shape):
    """MLX-aware shape computation."""
    if self.wiring:
        return self.wiring.compute_output_shape(input_shape)
    return self._compute_output_shape(input_shape)
```

### 3. State Handling
```python
def get_initial_state(self, inputs=None, batch_size=None):
    """MLX-native state initialization."""
    if inputs is not None:
        batch_size = inputs.shape[0]
    return mx.zeros((batch_size, self.state_size))
```

## Implementation Strategy

### 1. Core Functionality
- Wrap MLX operations
- Maintain MLX's performance
- Add NCP-specific features

### 2. Layer Types
- Dense layers
- RNN cells
- Custom NCP layers

### 3. Integration Points
- Wiring patterns
- State management
- Shape inference

## Usage Examples

### 1. Basic Layer
```python
layer = NCPDense(32, activation="tanh")
output = layer(inputs)  # MLX-native operation
```

### 2. RNN Cell
```python
cell = NCPRNNCell(64)
output, new_state = cell(inputs, states)
```

### 3. With Wiring
```python
layer = NCPDense(32, wiring=wiring_pattern)
output = layer(inputs)  # Applies wiring pattern
```

## Technical Considerations

### 1. Performance
- Minimize overhead
- Optimize MLX integration
- Efficient state management

### 2. Compatibility
- MLX version support
- Framework integration
- Future backend support

### 3. Extensibility
- Custom layer support
- Wiring pattern integration
- State customization

## Next Steps

1. Implementation Priority
   - Core layer functionality
   - RNN cell support
   - Wiring integration

2. Documentation
   - MLX-specific details
   - Usage patterns
   - Performance considerations

3. Testing
   - Performance benchmarks
   - Compatibility tests
   - Integration validation

## Questions to Address

1. Layer Configuration
   - How to handle MLX-specific parameters?
   - What configuration options to expose?
   - How to maintain MLX's flexibility?

2. State Management
   - How to optimize state handling?
   - What state patterns to support?
   - How to handle custom states?

3. Performance
   - Where to optimize MLX integration?
   - How to minimize overhead?
   - What operations to prioritize?