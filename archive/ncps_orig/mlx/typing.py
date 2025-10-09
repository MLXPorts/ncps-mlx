"""Type definitions for MLX neural circuit components.

This module provides common type aliases and definitions used throughout the MLX 
neural circuit implementations to improve code clarity and type safety.
"""

from typing import Union, Optional, List, Tuple, Dict, Any, Callable
import mlx.core as mx

# Common type aliases
Tensor = mx.array
"""Basic tensor type representing an MLX array"""

TensorOrFloat = Union[Tensor, float]
"""Type that can be either a tensor or float scalar"""

OptionalTensor = Optional[Tensor]
"""Optional tensor type"""

InitializerCallable = Callable[[Tensor], Tensor]
"""Type for weight initialization functions"""

ActivationCallable = Callable[[Tensor], Tensor]
"""Type for activation functions"""

# RNN specific types
BatchOrSingle = Union[Tuple[int, int], Tuple[int, ...]]
"""Shape type that can be either batched or single sample"""

RNNStates = List[Tensor]
"""Type representing RNN hidden states"""

RNNActivation = Union[str, ActivationCallable]
"""Type for RNN activation specifications"""

TimeSteps = Union[float, Tensor]
"""Type representing time step information"""

RNNOutput = Union[Tensor, Tuple[Tensor, RNNStates]]
"""Type for RNN outputs which may include state"""

# State dictionary types
StateDict = Dict[str, Any]
"""Type for model state dictionaries"""

Parameters = Dict[str, Union[Tensor, Dict[str, Tensor]]]
"""Type for model parameters"""

__all__ = [
    "Tensor",
    "TensorOrFloat", 
    "OptionalTensor",
    "InitializerCallable",
    "ActivationCallable",
    "BatchOrSingle",
    "RNNStates",
    "RNNActivation", 
    "TimeSteps",
    "RNNOutput",
    "StateDict",
    "Parameters",
]