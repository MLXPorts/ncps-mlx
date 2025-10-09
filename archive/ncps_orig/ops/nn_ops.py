"""Neural network specific operations."""

from typing import Optional, Union, Sequence
import numpy as np

from . import math_ops

# Type aliases
ArrayLike = Union[np.ndarray, float, int]


def sigmoid(x: ArrayLike) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def tanh(x: ArrayLike) -> np.ndarray:
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def relu(x: ArrayLike) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(x, 0)


def leaky_relu(x: ArrayLike, alpha: float = 0.2) -> np.ndarray:
    """Leaky Rectified Linear Unit activation function."""
    return np.where(x > 0, x, alpha * x)


def elu(x: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu(x: ArrayLike) -> np.ndarray:
    """Scaled Exponential Linear Unit activation function."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softplus(x: ArrayLike) -> np.ndarray:
    """Softplus activation function."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softsign(x: ArrayLike) -> np.ndarray:
    """Softsign activation function."""
    return x / (1 + np.abs(x))


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Log softmax activation function."""
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def dropout(
    x: ArrayLike,
    rate: float,
    training: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """Apply dropout to input.
    
    Args:
        x: Input array
        rate: Dropout rate (fraction of units to drop)
        training: Whether in training mode
        seed: Optional random seed
        
    Returns:
        Array with dropout applied
    """
    if not training or rate == 0:
        return x
        
    if seed is not None:
        np.random.seed(seed)
        
    keep_prob = 1.0 - rate
    # Generate random mask with correct shape
    mask = np.random.binomial(1, keep_prob, size=x.shape).astype(x.dtype)
    # Scale output to maintain expected value
    return x * mask / keep_prob


def batch_normalization(
    x: ArrayLike,
    mean: Optional[ArrayLike] = None,
    variance: Optional[ArrayLike] = None,
    offset: Optional[ArrayLike] = None,
    scale: Optional[ArrayLike] = None,
    variance_epsilon: float = 1e-3,
    training: bool = False
) -> np.ndarray:
    """Apply batch normalization."""
    x = np.asarray(x)
    
    if training or mean is None:
        mean = np.mean(x, axis=0)
    if training or variance is None:
        variance = np.var(x, axis=0)
        
    inv = 1.0 / np.sqrt(variance + variance_epsilon)
    normalized = (x - mean) * inv
    
    if scale is not None:
        normalized *= scale
    if offset is not None:
        normalized += offset
        
    return normalized


def layer_normalization(
    x: ArrayLike,
    offset: Optional[ArrayLike] = None,
    scale: Optional[ArrayLike] = None,
    axis: int = -1,
    epsilon: float = 1e-3
) -> np.ndarray:
    """Apply layer normalization."""
    x = np.asarray(x)
    
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    
    inv = 1.0 / np.sqrt(variance + epsilon)
    normalized = (x - mean) * inv
    
    if scale is not None:
        normalized *= scale
    if offset is not None:
        normalized += offset
        
    return normalized