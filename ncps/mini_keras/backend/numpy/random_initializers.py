import numpy as np

def random_normal(shape, mean=0.0, stddev=0.05, seed=None, dtype=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)

def truncated_normal(shape, mean=0.0, stddev=0.05, seed=None, dtype=None):
    if seed is not None:
        np.random.seed(seed)
    values = np.random.normal(loc=mean, scale=stddev, size=shape)
    values = np.clip(values, mean - 2 * stddev, mean + 2 * stddev)
    return values.astype(dtype)

def random_uniform(shape, minval=-0.05, maxval=0.05, seed=None, dtype=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low=minval, high=maxval, size=shape).astype(dtype)

def variance_scaling(shape, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None, dtype=None):
    fan_in, fan_out = compute_fans(shape)
    scale /= max(1.0, fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else (fan_in + fan_out) / 2.0)
    if distribution == "truncated_normal":
        stddev = np.sqrt(scale) / 0.87962566103423978
        return truncated_normal(shape, mean=0.0, stddev=stddev, seed=seed, dtype=dtype)
    elif distribution == "untruncated_normal":
        stddev = np.sqrt(scale)
        return random_normal(shape, mean=0.0, stddev=stddev, seed=seed, dtype=dtype)
    else:
        limit = np.sqrt(3.0 * scale)
        return random_uniform(shape, minval=-limit, maxval=limit, seed=seed, dtype=dtype)

def orthogonal(shape, gain=1.0, seed=None, dtype=None):
    if len(shape) < 2:
        raise ValueError("The tensor to initialize must be at least two-dimensional.")
    num_rows = np.prod(shape[:-1])
    num_cols = shape[-1]
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
    if seed is not None:
        np.random.seed(seed)
    a = np.random.normal(size=flat_shape).astype(dtype)
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    q *= np.sign(d)
    if num_rows < num_cols:
        q = q.T
    return gain * q[:num_rows, :num_cols].reshape(shape)

def compute_fans(shape):
    shape = tuple(shape)
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
