"""MLX-specific operations."""

import keras
from keras import ops as kops
import numpy as np
from typing import Optional, Union, Tuple, List, Any

# Types
Tensor = Any
TensorLike = Union[Tensor, np.ndarray, float, int]
Shape = Tuple[int, ...]
DType = str


def convert_to_tensor(
    x: TensorLike,
    dtype: Optional[DType] = None,
    sparse: bool = False
) -> Tensor:
    """Convert input to tensor."""
    return kops.convert_to_tensor(x, dtype=dtype)


def ones(shape: Shape, dtype: Optional[DType] = None) -> Tensor:
    """Create tensor of ones."""
    return kops.ones(shape, dtype=dtype)


def zeros(shape: Shape, dtype: Optional[DType] = None) -> Tensor:
    """Create tensor of zeros."""
    return kops.zeros(shape, dtype=dtype)


def ones_like(x: Tensor, dtype: Optional[DType] = None) -> Tensor:
    """Create tensor of ones with same shape as input."""
    return kops.ones_like(x, dtype=dtype)


def zeros_like(x: Tensor, dtype: Optional[DType] = None) -> Tensor:
    """Create tensor of zeros with same shape as input."""
    return kops.zeros_like(x, dtype=dtype)


def reshape(x: Tensor, shape: Shape) -> Tensor:
    """Reshape tensor."""
    return kops.reshape(x, shape)


def cast(x: Tensor, dtype: DType) -> Tensor:
    """Cast tensor to dtype."""
    return kops.cast(x, dtype)


def concatenate(tensors: List[Tensor], axis: int = -1) -> Tensor:
    """Concatenate tensors along axis."""
    return kops.concatenate(tensors, axis=axis)


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along new axis."""
    return kops.stack(tensors, axis=axis)


def split(x: Tensor, num_splits: int, axis: int = -1) -> List[Tensor]:
    """Split tensor into num_splits tensors along axis."""
    return kops.split(x, num_splits, axis=axis)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication."""
    return kops.matmul(a, b)


def transpose(x: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    """Transpose tensor."""
    return kops.transpose(x, axes)


def reduce_mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce mean along axis."""
    return kops.mean(x, axis=axis, keepdims=keepdims)


def reduce_sum(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce sum along axis."""
    return kops.sum(x, axis=axis, keepdims=keepdims)


def reduce_max(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce max along axis."""
    return kops.max(x, axis=axis, keepdims=keepdims)


def reduce_min(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce min along axis."""
    return kops.min(x, axis=axis, keepdims=keepdims)


def abs(x: Tensor) -> Tensor:
    """Absolute value."""
    return kops.abs(x)


def exp(x: Tensor) -> Tensor:
    """Exponential."""
    return kops.exp(x)


def log(x: Tensor) -> Tensor:
    """Natural logarithm."""
    return kops.log(x)


def pow(x: Tensor, y: Union[Tensor, float]) -> Tensor:
    """Power function."""
    return kops.pow(x, y)


def sqrt(x: Tensor) -> Tensor:
    """Square root."""
    return kops.sqrt(x)


def square(x: Tensor) -> Tensor:
    """Square."""
    return kops.square(x)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation."""
    return kops.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    """Tanh activation."""
    return kops.tanh(x)


def relu(x: Tensor) -> Tensor:
    """ReLU activation."""
    return kops.relu(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation."""
    return kops.softmax(x, axis=axis)


def clip(x: Tensor, min_value: float, max_value: float) -> Tensor:
    """Clip tensor values."""
    return kops.clip(x, min_value, max_value)


def maximum(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise maximum."""
    return kops.maximum(x, y)


def minimum(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise minimum."""
    return kops.minimum(x, y)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Where condition is True, yield x, otherwise yield y."""
    return kops.where(condition, x, y)


def all(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce all along axis."""
    return kops.all(x, axis=axis, keepdims=keepdims)


def any(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Reduce any along axis."""
    return kops.any(x, axis=axis, keepdims=keepdims)


def equal(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise equality."""
    return kops.equal(x, y)


def not_equal(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise inequality."""
    return kops.not_equal(x, y)


def greater(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greater than."""
    return kops.greater(x, y)


def greater_equal(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greater than or equal."""
    return kops.greater_equal(x, y)


def less(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise less than."""
    return kops.less(x, y)


def less_equal(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise less than or equal."""
    return kops.less_equal(x, y)