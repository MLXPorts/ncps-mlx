# Activation System Refactor Plan

## Overview
Refactor the activation system to use an abstract base class pattern instead of a backend approach. This will provide better type safety and clearer interface definitions for each backend implementation.

## Structure

```
ncps/
├── activations/
│   ├── __init__.py
│   └── activations.py  # Abstract base class definitions
└── mlx/
    └── activations/    # MLX-specific implementations
        ├── __init__.py
        └── activations.py
```

## Implementation Steps

1. Create Abstract Base Class
   - Define ActivationFunction ABC in ncps/activations/activations.py
   - Include abstract methods for common activation operations
   - Add type hints and docstrings for clear interface definition

2. Create MLX Implementation Directory
   - Create ncps/mlx/activations/ directory
   - Mirror the structure of base activations
   - Implement concrete activation classes for MLX

3. Abstract Methods to Include
   - `__call__`: Main activation function implementation
   - `gradient`: Gradient calculation if needed
   - Additional utility methods as needed

4. Example Implementation:

```python
# ncps/activations/activations.py
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        """Apply the activation function."""
        pass

    @abstractmethod
    def gradient(self, x):
        """Calculate the gradient of the activation function."""
        pass
```

```python
# ncps/mlx/activations/activations.py
from ncps.activations.activations import ActivationFunction
import mlx.core as mx

class MLXReLU(ActivationFunction):
    def __call__(self, x):
        return mx.maximum(0, x)
        
    def gradient(self, x):
        return mx.where(x > 0, 1.0, 0.0)
```

## Benefits
- Clear interface definition through ABC
- Type-safe implementations
- Better separation of concerns
- Easier to add new backends
- More maintainable code structure

## Next Steps
1. Implement base abstract classes
2. Create MLX activation implementations
3. Update existing code to use new structure
4. Add tests for new implementation
5. Update documentation