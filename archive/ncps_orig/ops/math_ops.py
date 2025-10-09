"""Mathematical operations."""

from typing import Optional, Union, Sequence
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, float, int]


def add(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise addition."""
    return np.add(x, y)


def subtract(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise subtraction."""
    return np.subtract(x, y)


def multiply(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise multiplication."""
    return np.multiply(x, y)


def divide(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise division."""
    return np.divide(x, y)


def matmul(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Matrix multiplication."""
    return np.matmul(x, y)


def reduce_mean(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Reduce mean along axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute mean
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Mean array
    """
    return np.mean(x, axis=axis, keepdims=keepdims)


def reduce_sum(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Reduce sum along axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute sum
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Sum array
    """
    return np.sum(x, axis=axis, keepdims=keepdims)


def reduce_max(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Reduce maximum along axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute maximum
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Maximum array
    """
    return np.max(x, axis=axis, keepdims=keepdims)


def reduce_min(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Reduce minimum along axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute minimum
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Minimum array
    """
    return np.min(x, axis=axis, keepdims=keepdims)


def abs(x: ArrayLike) -> np.ndarray:
    """Absolute value."""
    return np.abs(x)


def exp(x: ArrayLike) -> np.ndarray:
    """Exponential."""
    return np.exp(x)


def log(x: ArrayLike) -> np.ndarray:
    """Natural logarithm."""
    return np.log(x)


def pow(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Power function."""
    return np.power(x, y)


def sqrt(x: ArrayLike) -> np.ndarray:
    """Square root."""
    return np.sqrt(x)


def square(x: ArrayLike) -> np.ndarray:
    """Square."""
    return np.square(x)


def clip(x: ArrayLike, min_val: float, max_val: float) -> np.ndarray:
    """Clip values to range [min_val, max_val]."""
    return np.clip(x, min_val, max_val)


def maximum(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise maximum."""
    return np.maximum(x, y)


def minimum(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise minimum."""
    return np.minimum(x, y)


def negative(x: ArrayLike) -> np.ndarray:
    """Numerical negative."""
    return np.negative(x)


def sign(x: ArrayLike) -> np.ndarray:
    """Returns sign of values (-1, 0, or 1)."""
    return np.sign(x)


def floor(x: ArrayLike) -> np.ndarray:
    """Round down to nearest integer."""
    return np.floor(x)


def ceil(x: ArrayLike) -> np.ndarray:
    """Round up to nearest integer."""
    return np.ceil(x)


def round(x: ArrayLike) -> np.ndarray:
    """Round to nearest integer."""
    return np.round(x)


def mod(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise modulo."""
    return np.mod(x, y)


def reciprocal(x: ArrayLike) -> np.ndarray:
    """Element-wise reciprocal (1/x)."""
    return np.reciprocal(x)


def logical_and(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise logical AND."""
    return np.logical_and(x, y)


def logical_or(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise logical OR."""
    return np.logical_or(x, y)


def logical_not(x: ArrayLike) -> np.ndarray:
    """Element-wise logical NOT."""
    return np.logical_not(x)


def equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise equality comparison."""
    return np.equal(x, y)


def not_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise inequality comparison."""
    return np.not_equal(x, y)


def greater(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise greater than comparison."""
    return np.greater(x, y)


def greater_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise greater than or equal comparison."""
    return np.greater_equal(x, y)


def less(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise less than comparison."""
    return np.less(x, y)


def less_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise less than or equal comparison."""
    return np.less_equal(x, y)