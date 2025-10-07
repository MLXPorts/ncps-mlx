"""State management operations."""

from typing import Optional, Union, Sequence, Any
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[str, np.dtype]


class Variable:
    """Mutable variable wrapper."""
    
    def __init__(
        self,
        initial_value: ArrayLike,
        dtype: Optional[DType] = None,
        name: Optional[str] = None
    ):
        """Initialize variable.
        
        Args:
            initial_value: Initial value
            dtype: Optional dtype
            name: Optional name
        """
        self._value = np.asarray(initial_value, dtype=dtype)
        self._name = name
    
    @property
    def value(self) -> np.ndarray:
        """Get current value."""
        return self._value
    
    @property
    def dtype(self) -> np.dtype:
        """Get dtype."""
        return self._value.dtype
    
    @property
    def shape(self) -> tuple:
        """Get shape."""
        return self._value.shape
    
    @property
    def name(self) -> Optional[str]:
        """Get name."""
        return self._name


def assign(ref: Variable, value: ArrayLike) -> None:
    """Assign value to variable.
    
    Args:
        ref: Variable to update
        value: New value
    """
    ref._value = np.asarray(value, dtype=ref.dtype)


def assign_add(ref: Variable, value: ArrayLike) -> None:
    """Add value to variable.
    
    Args:
        ref: Variable to update
        value: Value to add
    """
    ref._value += np.asarray(value, dtype=ref.dtype)


def assign_sub(ref: Variable, value: ArrayLike) -> None:
    """Subtract value from variable.
    
    Args:
        ref: Variable to update
        value: Value to subtract
    """
    ref._value -= np.asarray(value, dtype=ref.dtype)


def scatter_update(
    ref: Variable,
    indices: ArrayLike,
    updates: ArrayLike,
    axis: Optional[int] = None
) -> None:
    """Update slices of variable.
    
    Args:
        ref: Variable to update
        indices: Indices to update
        updates: New values
        axis: Optional axis along which to index
    """
    if axis is None:
        ref._value[indices] = updates
    else:
        slices = [slice(None)] * ref._value.ndim
        slices[axis] = indices
        ref._value[tuple(slices)] = updates


def scatter_add(
    ref: Variable,
    indices: ArrayLike,
    updates: ArrayLike,
    axis: Optional[int] = None
) -> None:
    """Add updates to slices of variable.
    
    Args:
        ref: Variable to update
        indices: Indices to update
        updates: Values to add
        axis: Optional axis along which to index
    """
    if axis is None:
        ref._value[indices] += updates
    else:
        slices = [slice(None)] * ref._value.ndim
        slices[axis] = indices
        ref._value[tuple(slices)] += updates


def scatter_sub(
    ref: Variable,
    indices: ArrayLike,
    updates: ArrayLike,
    axis: Optional[int] = None
) -> None:
    """Subtract updates from slices of variable.
    
    Args:
        ref: Variable to update
        indices: Indices to update
        updates: Values to subtract
        axis: Optional axis along which to index
    """
    if axis is None:
        ref._value[indices] -= updates
    else:
        slices = [slice(None)] * ref._value.ndim
        slices[axis] = indices
        ref._value[tuple(slices)] -= updates


def scatter_mul(
    ref: Variable,
    indices: ArrayLike,
    updates: ArrayLike,
    axis: Optional[int] = None
) -> None:
    """Multiply slices of variable by updates.
    
    Args:
        ref: Variable to update
        indices: Indices to update
        updates: Values to multiply by
        axis: Optional axis along which to index
    """
    if axis is None:
        ref._value[indices] *= updates
    else:
        slices = [slice(None)] * ref._value.ndim
        slices[axis] = indices
        ref._value[tuple(slices)] *= updates


def scatter_div(
    ref: Variable,
    indices: ArrayLike,
    updates: ArrayLike,
    axis: Optional[int] = None
) -> None:
    """Divide slices of variable by updates.
    
    Args:
        ref: Variable to update
        indices: Indices to update
        updates: Values to divide by
        axis: Optional axis along which to index
    """
    if axis is None:
        ref._value[indices] /= updates
    else:
        slices = [slice(None)] * ref._value.ndim
        slices[axis] = indices
        ref._value[tuple(slices)] /= updates