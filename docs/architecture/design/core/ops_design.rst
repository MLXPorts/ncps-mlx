Operations Design
=================

Overview
--------

The operations module provides NumPy-based implementations of core
tensor operations needed for neural circuit policies.

Directory Structure
-------------------

::

ncps/ops/
    __init__.py          # Exports public API
    array_ops.py         # Basic array operations
    math_ops.py          # Mathematical operations
    nn_ops.py           # Neural network specific operations
    random_ops.py        # Random number generation
    state_ops.py         # State management operations

Module Descriptions
-------------------

array_ops.py
~~~~~~~~~~~~

Basic array manipulation operations: - reshape - concatenate - stack -
split - transpose - expand_dims - squeeze

math_ops.py
~~~~~~~~~~~

Mathematical operations: - add, subtract, multiply, divide - matmul -
reduce_mean, reduce_sum - reduce_max, reduce_min - abs, exp, log - pow,
sqrt, square - clip, maximum, minimum

nn_ops.py
~~~~~~~~~

Neural network specific operations: - sigmoid - tanh - relu - softmax -
dropout - batch_normalization

random_ops.py
~~~~~~~~~~~~~

Random number generation: - normal - uniform - bernoulli - seed
management

state_ops.py
~~~~~~~~~~~~

State management operations: - assign - scatter_update - gather

Implementation Guidelines
-------------------------

1. Use NumPy directly:

.. code:: python

import numpy as np

def matmul(a, b):
    """Matrix multiplication."""
    return np.matmul(a, b)

2. Maintain consistent interfaces:

.. code:: python

def reduce_mean(x, axis=None, keepdims=False):
    """Reduce mean along axis."""
    return np.mean(x, axis=axis, keepdims=keepdims)

3. Type hints and documentation:

.. code:: python

from typing import Optional, Union, Tuple
import numpy as np

def reshape(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape array to new shape.

    Args:
        x: Input array
        shape: New shape

    Returns:
        Reshaped array
    """
    return np.reshape(x, shape)

Usage Example
-------------

.. code:: python

from ncps.ops import array_ops, math_ops, nn_ops

# Create arrays
x = array_ops.ones((2, 3))
y = array_ops.zeros((3, 4))

# Perform operations
z = math_ops.matmul(x, y)
output = nn_ops.sigmoid(z)

MLX Integration
---------------

MLX-specific implementations will live in ``ncps.mlx.ops`` with the same
structure but using MLX’s array operations instead of NumPy.

Testing Strategy
----------------

1. Unit tests for each operation
2. Property-based tests for mathematical properties
3. Numerical stability tests
4. Edge case handling
5. Shape validation

Migration Plan
--------------

1. Create ops directory structure
2. Move operations from ops.py to appropriate modules
3. Update imports in existing code
4. Add comprehensive tests
5. Remove ops.py
