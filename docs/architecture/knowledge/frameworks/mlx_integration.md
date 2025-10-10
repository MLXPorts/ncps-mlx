# MLX Integration Design

## Overview

This document outlines the strategy for integrating MLX support into the Neural Circuit Policies framework. The goal is to provide efficient MLX-based implementations while maintaining the clean separation between core functionality and backend-specific code.

## Directory Structure

```
ncps/
  ops/              # NumPy-based core operations
    __init__.py
    array_ops.py
    math_ops.py
    nn_ops.py
    random_ops.py
    state_ops.py
    
  mlx/             # MLX-specific implementations
    ops/           # MLX operation implementations
      __init__.py
      array_ops.py
      math_ops.py
      nn_ops.py
      random_ops.py
      state_ops.py
      
    layers/        # MLX-specific layer implementations
      __init__.py
      layer.py     # Base MLX layer
      rnn.py       # MLX RNN implementation
      liquid.py    # MLX liquid neurons
```

## Implementation Strategy

### 1. Operations Layer

The MLX operations should mirror the NumPy-based ops interface but leverage MLX's capabilities:

```python
# ncps/mlx/ops/array_ops.py
import mlx.core as mx

def convert_to_tensor(x, dtype=None):
    """Convert input to MLX array."""
    return mx.array(x, dtype=dtype)

def matmul(a, b):
    """Matrix multiplication using MLX."""
    return mx.matmul(a, b)
```

Key considerations:
- Match NumPy ops signatures exactly
- Leverage MLX's automatic differentiation
- Use MLX's optimized operations where available
- Handle device placement (CPU/GPU)

### 2. Layer System

MLX-specific layer implementations now inherit directly from `mlx.nn.Module`:

```python
from mlx import nn

class Projector(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.linear = nn.Linear(in_dims, out_dims)
    
    def __call__(self, x):
        return nn.relu(self.linear(x))
```

Benefits:
- Leverages native MLX parameter/state management
- Keeps the code aligned with upstream APIs
- Reduces maintenance overhead

### 3. RNN Implementation

The MLX RNN implementation should leverage MLX's capabilities:

```python
class RNN(nn.Module):
    def __init__(self, cell, return_sequences=False):
        super().__init__()
        self.cell = cell
        self.return_sequences = return_sequences
        
    def __call__(self, inputs):
        return self.cell(inputs)
```

Features:
- Efficient sequence processing
- MLX-optimized state management
- Support for MLX's automatic batching

### 4. Liquid Neurons

MLX-specific liquid neuron implementations:

Liquid neuron implementations continue to subclass `mlx.nn.Module`, ensuring\n+compatibility with MLX optimizers and training utilities without extra shims.

## Testing Strategy

1. Unit Tests:
   - Mirror NumPy tests for MLX implementations
   - Add MLX-specific test cases
   - Test device placement and optimization

2. Integration Tests:
   - End-to-end MLX model tests
   - Performance comparisons
   - Memory usage validation

3. Benchmark Suite:
   - Compare MLX vs NumPy performance
   - Measure optimization impacts
   - Profile memory usage

## Migration Guide

1. For Users:
   ```python
   # NumPy-based usage
   from ncps import ops
   
   # MLX-based usage
   from ncps.mlx import ops
   ```

2. For Developers:
   - Follow parallel implementation pattern
   - Maintain consistent interfaces
   - Document MLX-specific features

## Performance Considerations

1. MLX Optimizations:
   - Use MLX's lazy evaluation
   - Leverage automatic differentiation
   - Optimize memory access patterns

2. Device Management:
   - Handle CPU/GPU placement
   - Manage memory transfers
   - Support device-specific optimizations

3. Batching Strategy:
   - Use MLX's automatic batching
   - Optimize sequence processing
   - Handle variable-length sequences

## Next Steps

1. Implementation Order:
   - Core MLX ops implementation
   - Base MLX layer system
   - RNN and cell implementations
   - Liquid neuron implementations

2. Testing:
   - Unit test suite
   - Integration tests
   - Performance benchmarks

3. Documentation:
   - API documentation
   - Migration guides
   - Performance tips

4. Optimization:
   - Profile and optimize
   - Add MLX-specific features
   - Performance tuning
