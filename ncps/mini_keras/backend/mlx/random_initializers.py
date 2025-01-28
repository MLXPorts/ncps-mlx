import mlx.core as mx

def random_normal(shape, mean=0.0, stddev=0.05, seed=None, dtype=None):
    return mx.random.normal(shape, mean=mean, stddev=stddev, seed=seed, dtype=dtype)

def truncated_normal(shape, mean=0.0, stddev=0.05, seed=None, dtype=None):
    return mx.random.truncated_normal(shape, mean=mean, stddev=stddev, seed=seed, dtype=dtype)

def random_uniform(shape, minval=-0.05, maxval=0.05, seed=None, dtype=None):
    return mx.random.uniform(shape, minval=minval, maxval=maxval, seed=seed, dtype=dtype)

def variance_scaling(shape, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None, dtype=None):
    fan_in, fan_out = compute_fans(shape)
    scale = mx.divide(scale, mx.maximum(1.0, fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else (fan_in + fan_out) / 2.0))
    if distribution == "truncated_normal":
        stddev = mx.divide(mx.sqrt(scale), 0.87962566103423978)
        return mx.random.truncated_normal(shape, mean=0.0, stddev=stddev, seed=seed, dtype=dtype)
    elif distribution == "untruncated_normal":
        stddev = mx.sqrt(scale)
        return mx.random.normal(shape, mean=0.0, stddev=stddev, seed=seed, dtype=dtype)
    else:
        limit = mx.sqrt(3.0 * scale)
        return mx.random.uniform(shape, minval=-limit, maxval=limit, seed=seed, dtype=dtype)

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
        receptive_field_size = mx.prod(mx.array(shape[:-2]))
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
