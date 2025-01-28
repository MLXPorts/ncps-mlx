import mlx.core as mx
from ncps.mini_keras.backend import floatx
from ncps.mini_keras.backend.mlx.nn import softmax
from ncps.mini_keras.random.seed_generator import SeedGenerator
from ncps.mini_keras.random.seed_generator import draw_seed
from ncps.mini_keras.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    # Use MLX's random normal
    return mx.random.normal(shape, mean=mean, std=stddev, dtype=dtype, key=seed)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    # Use MLX's random uniform
    return mx.random.uniform(shape, low=minval, high=maxval, dtype=dtype, key=seed)


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = draw_seed(seed)
    output = []
    for logits_instance in logits:
        probabilities = softmax(logits_instance)
        classes = mx.arange(logits_instance.shape[-1])
        samples = mx.random.choice(
            classes, 
            shape=(num_samples,), 
            p=probabilities, 
            key=seed
        )
        output.append(samples)
    return mx.stack(output).astype(dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = draw_seed(seed)
    # Use MLX's random integers
    return mx.random.randint(minval, maxval, shape=shape, dtype=dtype, key=seed)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    # Use rejection sampling with MLX's random normal
    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev
    
    result = mx.random.normal(shape, mean=mean, std=stddev, dtype=dtype, key=seed)
    while True:
        invalid_mask = (result < lower_bound) | (result > upper_bound)
        if not mx.any(invalid_mask):
            break
        new_values = mx.random.normal(
            shape, 
            mean=mean, 
            std=stddev, 
            dtype=dtype, 
            key=seed
        )
        result = mx.where(invalid_mask, new_values, result)
    
    return result


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    keep_prob = 1.0 - rate

    if noise_shape is None:
        noise_shape = inputs.shape
    
    mask = mx.random.uniform(noise_shape, key=seed) < keep_prob
    mask = mx.broadcast_to(mask, inputs.shape)
    return mx.where(mask, inputs / keep_prob, mx.zeros_like(inputs))


def shuffle(x, axis=0, seed=None):
    seed = draw_seed(seed)
    indices = mx.random.permutation(x.shape[axis], key=seed)
    return mx.take(x, indices, axis=axis)


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = mx.random.default_rng(seed)
    return rng.gamma(alpha, scale=1.0, size=shape).astype(dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = mx.random.default_rng(seed)
    sample = rng.binomial(n=counts, p=probabilities, size=shape).astype(dtype)
    return sample


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = mx.random.default_rng(seed)
    sample = rng.beta(a=alpha, b=beta, size=shape).astype(dtype)
    return sample
