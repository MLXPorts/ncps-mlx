"""Comparison operations."""

import numpy as np
from typing import Optional, Union, Sequence

# Type aliases
ArrayLike = Union[np.ndarray, float, int]


def all(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Test whether all array elements along a given axis evaluate to True.
    
    Args:
        x: Input array
        axis: Axis along which to perform operation
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Boolean array
    """
    return np.all(x, axis=axis, keepdims=keepdims)


def any(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """Test whether any array elements along a given axis evaluate to True.
    
    Args:
        x: Input array
        axis: Axis along which to perform operation
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Boolean array
    """
    return np.any(x, axis=axis, keepdims=keepdims)


def equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise equality comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.equal(x, y)


def not_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise inequality comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.not_equal(x, y)


def greater(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise greater than comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.greater(x, y)


def greater_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise greater than or equal comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.greater_equal(x, y)


def less(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise less than comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.less(x, y)


def less_equal(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise less than or equal comparison.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.less_equal(x, y)


def logical_and(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise logical AND.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.logical_and(x, y)


def logical_or(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Element-wise logical OR.
    
    Args:
        x: First input array
        y: Second input array
        
    Returns:
        Boolean array
    """
    return np.logical_or(x, y)


def logical_not(x: ArrayLike) -> np.ndarray:
    """Element-wise logical NOT.
    
    Args:
        x: Input array
        
    Returns:
        Boolean array
    """
    return np.logical_not(x)


def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean array
        x: Values to use where condition is True
        y: Values to use where condition is False
        
    Returns:
        Array with values from x where condition is True, y otherwise
    """
    return np.where(condition, x, y)