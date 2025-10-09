"""MLX array operations."""

from typing import Optional, Union, Sequence, Tuple, List
import mlx.core as mx
import numpy as np

# Type aliases
ArrayLike = Union[mx.array, np.ndarray, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[str, np.dtype]


def convert_to_tensor(
    x: ArrayLike,
    dtype: Optional[DType] = None,
    device: Optional[str] = None
) -> mx.array:
    """Convert input to MLX array."""
    # Handle numpy arrays
    if isinstance(x, np.ndarray):
        x = mx.array(x, dtype=dtype)
    # Handle python scalars
    elif isinstance(x, (float, int)):
        x = mx.array([x], dtype=dtype)
    # Handle sequences
    elif isinstance(x, (list, tuple)):
        x = mx.array(x, dtype=dtype)
    # Handle MLX arrays
    elif isinstance(x, mx.array):
        if dtype is not None:
            x = mx.astype(x, dtype)
    else:
        raise ValueError(f"Cannot convert type {type(x)} to MLX array")
    
    # Handle device placement
    if device is not None:
        x = x.to(device)
    
    return x


def zeros(
    shape: Shape,
    dtype: Optional[DType] = None,
    device: Optional[str] = None
) -> mx.array:
    """Create array of zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    x = mx.zeros(shape, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def ones(
    shape: Shape,
    dtype: Optional[DType] = None,
    device: Optional[str] = None
) -> mx.array:
    """Create array of ones."""
    if isinstance(shape, int):
        shape = (shape,)
    x = mx.ones(shape, dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def zeros_like(
    x: ArrayLike,
    dtype: Optional[DType] = None,
    device: Optional[str] = None
) -> mx.array:
    """Create array of zeros with same shape as input."""
    x = convert_to_tensor(x)
    return zeros(x.shape, dtype=dtype or x.dtype, device=device)


def ones_like(
    x: ArrayLike,
    dtype: Optional[DType] = None,
    device: Optional[str] = None
) -> mx.array:
    """Create array of ones with same shape as input."""
    x = convert_to_tensor(x)
    return ones(x.shape, dtype=dtype or x.dtype, device=device)


def reshape(x: ArrayLike, shape: Shape) -> mx.array:
    """Reshape array to new shape."""
    x = convert_to_tensor(x)
    if isinstance(shape, int):
        shape = (shape,)
    return mx.reshape(x, shape)


def concatenate(arrays: Sequence[ArrayLike], axis: int = -1) -> mx.array:
    """Concatenate arrays along axis."""
    arrays = [convert_to_tensor(x) for x in arrays]
    return mx.concatenate(arrays, axis=axis)


def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> mx.array:
    """Stack arrays along new axis."""
    arrays = [convert_to_tensor(x) for x in arrays]
    return mx.stack(arrays, axis=axis)


def split(
    x: ArrayLike,
    num_or_size_splits: Union[int, Sequence[int]],
    axis: int = 0
) -> List[mx.array]:
    """Split array into multiple sub-arrays."""
    x = convert_to_tensor(x)
    dim_size = x.shape[axis]
    
    if isinstance(num_or_size_splits, int):
        # Equal splits
        split_size = dim_size // num_or_size_splits
        sizes = [split_size] * num_or_size_splits
    else:
        # Size-based splits
        sizes = list(num_or_size_splits)
        if sum(sizes) != dim_size:
            raise ValueError(
                f"Sum of sizes {sum(sizes)} does not match dim size {dim_size}"
            )
    
    # Calculate split points
    split_points = []
    current_point = 0
    for size in sizes[:-1]:
        current_point += size
        split_points.append(current_point)
    
    # Split array
    parts = mx.split(x, split_points, axis=axis)
    
    # Verify split sizes
    for part, size in zip(parts, sizes):
        expected_shape = list(x.shape)
        expected_shape[axis] = size
        assert part.shape == tuple(expected_shape)
    
    return parts


def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> mx.array:
    """Permute array dimensions."""
    x = convert_to_tensor(x)
    return mx.transpose(x, axes)


def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> mx.array:
    """Insert new axes into array shape."""
    x = convert_to_tensor(x)
    if isinstance(axis, (list, tuple)):
        for a in sorted(axis):
            x = mx.expand_dims(x, a)
        return x
    return mx.expand_dims(x, axis)


def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> mx.array:
    """Remove single-dimensional entries from array shape."""
    x = convert_to_tensor(x)
    return mx.squeeze(x, axis)


def tile(x: ArrayLike, reps: Union[int, Sequence[int]]) -> mx.array:
    """Construct array by repeating x the number of times given by reps."""
    x = convert_to_tensor(x)
    if isinstance(reps, int):
        reps = (reps,)
    return mx.tile(x, reps)


def pad(
    x: ArrayLike,
    pad_width: Union[int, Sequence[Sequence[int]]],
    mode: str = "constant",
    constant_values: Union[int, float] = 0
) -> mx.array:
    """Pad array."""
    x = convert_to_tensor(x)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * len(x.shape)
    return mx.pad(x, pad_width, mode=mode, constant_values=constant_values)


def slice(
    x: ArrayLike,
    begin: Union[int, Sequence[int]],
    size: Optional[Union[int, Sequence[int]]] = None
) -> mx.array:
    """Extract slice from array.
    
    Args:
        x: Input array
        begin: Starting indices
        size: Optional size of each slice dimension
        
    Returns:
        Sliced array
    """
    x = convert_to_tensor(x)
    
    # Handle scalar inputs
    if isinstance(begin, (int, np.integer)):
        begin = [begin]
    elif begin is None:
        raise ValueError("begin cannot be None")
    else:
        begin = list(begin)
    
    # Handle size
    if size is not None:
        if isinstance(size, (int, np.integer)):
            size = [size]
        size = list(size)
    else:
        size = [None] * len(begin)
    
    # Create slice objects
    slices = []
    for b, s in zip(begin, size):
        if s is None:
            slices.append(slice(b, None))
        else:
            slices.append(slice(b, b + s))
    
    return x[tuple(slices)]


def gather(
    x: ArrayLike,
    indices: ArrayLike,
    axis: Optional[int] = None,
    batch_dims: int = 0
) -> mx.array:
    """Gather slices from x according to indices."""
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    
    # Default axis is 0
    if axis is None:
        axis = 0
        
    # Handle negative axis
    if axis < 0:
        axis = len(x.shape) + axis
        
    # Compute output shape
    out_shape = list(x.shape)
    out_shape[axis] = indices.shape[0]
    
    # Gather values
    result = mx.take(x, indices, axis=axis)
    
    # Reshape to correct output shape if needed
    if result.shape != tuple(out_shape):
        result = mx.reshape(result, out_shape)
    
    return result