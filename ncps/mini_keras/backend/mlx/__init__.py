"""MLX backend implementation for mini-keras.

This module provides MLX-specific implementations of core Keras operations
and utilities, enabling the use of Apple's MLX framework as a backend.
"""

from ncps.mini_keras.backend.common.name_scope import name_scope
from ncps.mini_keras.backend.mlx.core import (
    IS_THREAD_SAFE,
    SUPPORTS_RAGGED_TENSORS,
    SUPPORTS_SPARSE_TENSORS,
    Variable,
    cast,
    compute_output_spec,
    cond,
    convert_to_numpy,
    convert_to_tensor,
    device_scope,
    is_tensor,
    random_seed_dtype,
    shape,
    vectorized_map,
)

# Domain-specific operations
from ncps.mini_keras.backend.mlx import (
    core,
    image,
    linalg,
    math,
    nn,
    random,
    orthogonal,
    hpc,
    ode,
    random,
    numpy,
    layer,
    distribution_lib,
)

# RNN operations
from ncps.mini_keras.backend.mlx.rnn import (
    cudnn_ok,
    gru,
    lstm,
    rnn,
)

__all__ = [
    # Core attributes
    "IS_THREAD_SAFE",
    "SUPPORTS_RAGGED_TENSORS", 
    "SUPPORTS_SPARSE_TENSORS",
    # Core classes
    "Variable",
    # Core functions
    "name_scope",
    "cast",
    "compute_output_spec",
    "cond",
    "convert_to_numpy",
    "convert_to_tensor",
    "device_scope",
    "is_tensor",
    "random_seed_dtype",
    "shape",
    "vectorized_map",
    "ode",
    "numpy",
    "random",
    "core",
    "image", 
    "linalg",
    "math",
    "nn",
    "random",
    "orthogonal",
    "hpc",
    "distribution_lib",
    # RNN functions
    "cudnn_ok",
    "gru",
    "lstm", 
    "rnn",
    "layer",
]
