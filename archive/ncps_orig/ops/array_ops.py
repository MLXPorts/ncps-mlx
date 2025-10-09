"""Array manipulation operations."""

from typing import Optional, Union, Sequence, Tuple
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[str, np.dtype]


def convert_to_tensor(x: ArrayLike, dtype: Optional[DType] = None) -> np.ndarray:
    """Convert input to numpy array."""
    return np.asarray(x, dtype=dtype)


def zeros(shape: Shape, dtype: Optional[DType] = None) -> np.ndarray:
    """Create array of zeros.
    
    Args:
        shape: Array shape
        dtype: Optional dtype
        
    Returns:
        Array of zeros
    """
    return np.zeros(shape, dtype=dtype)


def ones(shape: Shape, dtype: Optional[DType] = None) -> np.ndarray:
    """Create array of ones.
    
    Args:
        shape: Array shape
        dtype: Optional dtype
        
    Returns:
        Array of ones
    """
    return np.ones(shape, dtype=dtype)


def zeros_like(x: ArrayLike, dtype: Optional[DType] = None) -> np.ndarray:
    """Create array of zeros with same shape as input."""
    return np.zeros_like(x, dtype=dtype)


def ones_like(x: ArrayLike, dtype: Optional[DType] = None) -> np.ndarray:
    """Create array of ones with same shape as input."""
    return np.ones_like(x, dtype=dtype)


def reshape(x: ArrayLike, shape: Shape) -> np.ndarray:
    """Reshape array to new shape."""
    return np.reshape(x, shape)


def concatenate(arrays: Sequence[ArrayLike], axis: int = -1) -> np.ndarray:
    """Concatenate arrays along axis."""
    return np.concatenate(arrays, axis=axis)


def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """Stack arrays along new axis."""
    return np.stack(arrays, axis=axis)


def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> Sequence[np.ndarray]:
    """Split array into multiple sub-arrays."""
    return np.split(x, num_or_size_splits, axis=axis)


def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Permute array dimensions."""
    return np.transpose(x, axes)


def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> np.ndarray:
    """Insert new axes into array shape."""
    return np.expand_dims(x, axis)


def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
    """Remove single-dimensional entries from array shape."""
    return np.squeeze(x, axis)


def tile(x: ArrayLike, reps: Union[int, Sequence[int]]) -> np.ndarray:
    """Construct array by repeating x the number of times given by reps."""
    return np.tile(x, reps)


def pad(
    x: ArrayLike,
    pad_width: Union[int, Sequence[Sequence[int]]],
    mode: str = "constant",
    constant_values: Union[int, float] = 0
) -> np.ndarray:
    """Pad array."""
    return np.pad(x, pad_width, mode=mode, constant_values=constant_values)


def slice(
    x: ArrayLike,
    begin: Sequence[int],
    size: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Extract slice from array."""
    if size is None:
        size = [None] * len(begin)
    slices = tuple(slice(b, None if s is None else b + s) for b, s in zip(begin, size))
    return x[slices]


def gather(
    x: ArrayLike,
    indices: ArrayLike,
    axis: Optional[int] = None,
    batch_dims: int = 0
) -> np.ndarray:
    """Gather slices from x according to indices."""
    if axis is None:
        axis = 0
    if batch_dims == 0:
        return np.take(x, indices, axis=axis)
    else:
        # Handle batch dimensions
        batch_shape = x.shape[:batch_dims]
        gather_shape = indices.shape[batch_dims:]
        result_shape = batch_shape + gather_shape + x.shape[axis+1:]
        
        # Reshape for batched gathering
        x_flat = x.reshape((-1,) + x.shape[batch_dims:])
        indices_flat = indices.reshape((-1,) + indices.shape[batch_dims:])
        
        # Compute offsets for each batch
        multiplier = np.prod(x.shape[batch_dims:axis+1])
        offsets = np.arange(len(x_flat)) * multiplier
        
        # Add offsets to indices
        indices_offset = indices_flat + offsets.reshape((-1,) + (1,) * len(indices.shape[batch_dims:]))
        
        # Gather and reshape
        result = np.take(x_flat, indices_offset, axis=axis)
        return result.reshape(result_shape)