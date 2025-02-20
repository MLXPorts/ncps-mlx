"""MLX operations for Neural Circuit Policies."""

from ncps.mlx.ops.array_ops import (
    convert_to_tensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    reshape,
    concatenate,
    stack,
    split,
    transpose,
    expand_dims,
    squeeze,
    tile,
    pad,
    slice,
    gather
)

# Version information
__version__ = '0.1.0'

# Set module docstring
__doc__ = """MLX operations for Neural Circuit Policies.

This module provides MLX-based implementations of core tensor operations
needed for neural circuit policies. The operations are organized into
several categories:

- Array operations (reshape, concatenate, etc.)
- Mathematical operations (add, matmul, etc.)
- Neural network operations (sigmoid, dropout, etc.)
- Random number generation (normal, uniform, etc.)
- State management (Variable, assign, etc.)

Example usage:
    >>> from ncps.mlx import ops
    >>> x = ops.ones((2, 3))
    >>> y = ops.matmul(x, ops.transpose(x))
    >>> z = ops.sigmoid(y)
"""