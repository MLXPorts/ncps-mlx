"""Random number generation operations."""

from typing import Optional, Union, Sequence, Tuple
import numpy as np

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Union[str, np.dtype]


def set_seed(seed: int) -> None:
    """Set random seed."""
    np.random.seed(seed)


def get_rng_state() -> Tuple:
    """Get random number generator state."""
    return np.random.get_state()


def set_rng_state(state: Tuple) -> None:
    """Set random number generator state."""
    np.random.set_state(state)


def normal(
    shape: Shape,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random normal values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, stddev, size=shape).astype(dtype)


def uniform(
    shape: Shape,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random uniform values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(minval, maxval, size=shape).astype(dtype)


def truncated_normal(
    shape: Shape,
    mean: float = 0.0,
    stddev: float = 1.0,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random truncated normal values."""
    if seed is not None:
        np.random.seed(seed)
        
    if minval is None:
        minval = mean - 2 * stddev
    if maxval is None:
        maxval = mean + 2 * stddev
        
    size = np.prod(shape) if isinstance(shape, (tuple, list)) else shape
    samples = np.random.normal(mean, stddev, size=size * 2)
    samples = samples[(samples >= minval) & (samples <= maxval)][:size]
    
    if isinstance(shape, (tuple, list)):
        samples = samples.reshape(shape)
        
    return samples.astype(dtype)


def bernoulli(
    shape: Shape,
    p: float = 0.5,  # Changed from prob to p to match test
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random bernoulli values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.binomial(1, p, size=shape).astype(dtype)


def gamma(
    shape: Shape,
    alpha: float,
    beta: float = 1.0,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random gamma values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.gamma(alpha, scale=1.0/beta, size=shape).astype(dtype)


def poisson(
    shape: Shape,
    lam: float,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random poisson values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.poisson(lam, size=shape).astype(dtype)


def shuffle(x: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Randomly shuffle array."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(x)


def random_crop(
    value: np.ndarray,
    size: Sequence[int],
    seed: Optional[int] = None
) -> np.ndarray:
    """Randomly crop array to size."""
    if seed is not None:
        np.random.seed(seed)
        
    input_shape = value.shape
    starts = [np.random.randint(0, s - c + 1) for s, c in zip(input_shape, size)]
    ends = [s + c for s, c in zip(starts, size)]
    slices = tuple(slice(s, e) for s, e in zip(starts, ends))
    
    return value[slices]