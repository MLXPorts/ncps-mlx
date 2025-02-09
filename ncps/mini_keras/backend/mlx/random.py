from ncps.mini_keras.random.seed_generator import draw_seed

import mlx.core as mx
from mlx.utils import tree_map

def draw_seed(seed=None):
    return mx.random.key(seed if seed is not None else mx.random.generate_seed())

def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    key = draw_seed(seed)
    return mx.random.normal(shape, mean, stddev, key=key).astype(dtype or mx.float32)

def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    key = draw_seed(seed)
    return mx.random.uniform(shape, minval, maxval, key=key).astype(dtype or mx.float32)

def categorical(logits, num_samples, dtype="int64", seed=None):
    key = draw_seed(seed)
    return mx.random.categorical(logits, num_samples, key=key).astype(dtype)

def randint(shape, minval, maxval, dtype="int32", seed=None):
    key = draw_seed(seed)
    return mx.random.randint(shape, minval, maxval, key=key).astype(dtype)

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    key = draw_seed(seed)
    samples = mx.random.normal(shape, mean, stddev, key=key)
    samples = mx.clip(samples, mean - 2*stddev, mean + 2*stddev)
    return samples.astype(dtype or mx.float32)

def dropout(inputs, rate, noise_shape=None, seed=None):
    key = draw_seed(seed)
    return mx.nn.dropout(inputs, rate, key=key)

def shuffle(x, axis=0, seed=None):
    key = draw_seed(seed)
    return mx.random.permutation(x, axis, key=key)

#########################
### Gamma Distribution ##
#########################

def gamma(shape, scale=1.0, num_candidates=10, key=None):
    """
    Gamma distribution sampler using Marsaglia-Tsang algorithm
    with shape >= 1 and auxiliary uniform sampling for shape < 1
    """
    key = key if key is not None else mx.random.key(0)
    shape = mx.array(shape)
    
    def _gamma_ge1(shape, scale, key):
        """Marsaglia-Tsang algorithm for shape >= 1"""
        d = shape - 1/3
        v = 1 / mx.sqrt(9 * d)
        
        key, ukey, zkey = mx.random.split(key, 3)
        U = mx.random.uniform(shape=(num_candidates,), key=ukey)
        Z = mx.random.normal(shape=(num_candidates,), key=zkey)
        
        t = (1 + v * Z)**3
        X = d * t
        log_accept = 0.5 * Z**2 + d * (1 - t + mx.log(t))
        accept_mask = mx.log(U) < log_accept
        
        valid = mx.argmax(accept_mask, axis=0)
        return X[valid] * scale

    def _gamma_lt1(shape, scale, key):
        """Auxiliary uniform method for shape < 1"""
        key, gkey, ukey = mx.random.split(key, 3)
        g = gamma(shape + 1, 1.0, num_candidates, gkey)
        u = mx.random.uniform(key=ukey)
        return g * u ** (1 / shape) * scale

    return mx.cond(
        shape >= 1,
        lambda k: _gamma_ge1(shape, scale, k),
        lambda k: _gamma_lt1(shape, scale, k),
        key
    )

##########################
### Binomial Distribution #
##########################

def binomial(n, p, shape=(), key=None):
    """
    Binomial distribution using vectorized Bernoulli trials
    Args:
        n: number of trials per sample
        p: probability of success
        shape: output shape (samples,)
    """
    key = key if key is not None else mx.random.key(0)
    trials = mx.random.bernoulli(p, shape + (n,), key=key)
    return mx.sum(trials.astype(mx.int32), axis=-1)

#######################
### Beta Distribution #
#######################

def beta(a, b, key=None):
    """Beta distribution using gamma variates"""
    key = key if key is not None else mx.random.key(0)
    key, gkey1, gkey2 = mx.random.split(key, 3)
    
    ga = gamma(a, 1.0, key=gkey1)
    gb = gamma(b, 1.0, key=gkey2)
    
    return ga / (ga + gb)
def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError("Gamma distribution not directly implemented in MLX")

def binomial(shape, counts, probabilities, dtype=None, seed=None):
    key = draw_seed(seed)
    return mx.random.bernoulli(probabilities, shape, key=key).astype(dtype or mx.float32)

def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("Beta distribution not directly implemented in MLX")
