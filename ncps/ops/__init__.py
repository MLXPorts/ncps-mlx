"""Neural Circuit Policies operations."""

from ncps.ops.array_ops import (
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

from ncps.ops.math_ops import (
    add,
    subtract,
    multiply,
    divide,
    matmul,
    reduce_mean,
    reduce_sum,
    reduce_max,
    reduce_min,
    abs,
    exp,
    log,
    pow,
    sqrt,
    square,
    clip,
    maximum,
    minimum,
    negative,
    sign,
    floor,
    ceil,
    round,
    mod,
    reciprocal
)

from ncps.ops.nn_ops import (
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    elu,
    selu,
    softplus,
    softsign,
    softmax,
    log_softmax,
    dropout,
    batch_normalization,
    layer_normalization
)

from ncps.ops.random_ops import (
    set_seed,
    get_rng_state,
    set_rng_state,
    normal,
    uniform,
    truncated_normal,
    bernoulli,
    gamma,
    poisson,
    shuffle,
    random_crop
)

from ncps.ops.comparison_ops import (
    all,
    any,
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    logical_and,
    logical_or,
    logical_not,
    where
)

# Version information
__version__ = '0.1.0'

# Set module docstring
__doc__ = """Neural Circuit Policies operations.

This module provides NumPy-based implementations of core tensor operations
needed for neural circuit policies. The operations are organized into
several categories:

- Array operations (reshape, concatenate, etc.)
- Mathematical operations (add, matmul, etc.)
- Neural network operations (sigmoid, dropout, etc.)
- Random number generation (normal, uniform, etc.)
- Comparison operations (equal, greater, etc.)

Example usage:
    >>> import ncps.ops as ops
    >>> x = ops.ones((2, 3))
    >>> y = ops.matmul(x, ops.transpose(x))
    >>> z = ops.sigmoid(y)
"""

""" # Clean up namespace
del array_ops
del math_ops
del nn_ops
del random_ops
del comparison_ops """