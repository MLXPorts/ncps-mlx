# Liquid Neural Network Utilities Design

## Overview
Core utilities and mixins for liquid neural network implementations, providing time handling, backbone networks, and activation functions.

## Components

### 1. Time Handling Mixin
```python
class TimeAwareMixin:
    """Mixin for handling time-based updates in liquid neurons."""
    
    def process_time_delta(self, time_delta, batch_size, seq_len):
        """Process time delta into consistent format."""
```

Key features:
- Handle None, float, and tensor time deltas
- Proper broadcasting for batch and sequence dimensions
- Consistent shape handling

### 2. Backbone Network Mixin
```python
class BackboneMixin:
    """Mixin for handling backbone networks in liquid neurons."""
    
    def build_backbone(self, input_size, units, layers, dropout):
        """Build backbone layers."""
        
    def apply_backbone(self, x, training=False):
        """Apply backbone to input."""
```

Key features:
- Flexible backbone architecture
- Proper dropout handling
- Training mode support

### 3. Activation Functions
```python
def lecun_tanh(x):
    """LeCun's tanh activation: 1.7159 * tanh(0.666 * x)"""
    
def get_activation(name):
    """Get activation function by name."""
```

Supported activations:
- lecun_tanh (scaled tanh)
- tanh
- relu
- gelu
- sigmoid

### 4. Utility Functions
```python
def broadcast_to_batch(x, batch_size):
    """Broadcast tensor to batch dimension."""
    
def ensure_time_dim(x):
    """Ensure tensor has time dimension."""
```

Common operations:
- Shape manipulation
- Dimension handling
- Type conversion

## Integration

### With Base Cell
```python
class BaseCell(keras.layers.Layer):
    def __init__(self):
        self.time_mixin = TimeAwareMixin()
        self.backbone_mixin = BackboneMixin()
```

### With RNN Layer
```python
class BaseRNN(keras.layers.Layer):
    def __init__(self):
        self.time_mixin = TimeAwareMixin()
```

## Key Differences from MLX Version

1. Keras Integration
   - Use Keras ops instead of MLX
   - Follow Keras layer patterns
   - Support Keras training loops

2. Simplified Design
   - Focus on liquid neuron needs
   - Remove unnecessary complexity
   - Better error messages

3. Enhanced Features
   - More flexible backbone options
   - Better time delta handling
   - Improved activation functions

## Implementation Steps

1. Core Utilities
   - Time handling functions
   - Backbone network support
   - Activation functions

2. Mixin Classes
   - TimeAwareMixin implementation
   - BackboneMixin implementation
   - Integration helpers

3. Testing
   - Unit tests for each component
   - Integration tests with cells
   - Performance validation

## Usage Examples

### Time Handling
```python
# Process time delta
time = self.process_time_delta(dt, batch_size, seq_len)

# Apply time scaling
output = self.apply_time_scale(state, time)
```

### Backbone Networks
```python
# Build backbone
self.backbone = self.build_backbone(input_size, units, layers)

# Apply backbone
features = self.apply_backbone(inputs, training=training)
```

### Activations
```python
# Get activation
activation = get_activation('lecun_tanh')

# Apply activation
output = activation(inputs)
```

## Benefits

1. Maintainability
   - Clear separation of concerns
   - Reusable components
   - Easy to extend

2. Performance
   - Efficient implementations
   - Optimized for Keras
   - Better memory usage

3. Usability
   - Simple API
   - Consistent behavior
   - Good error messages