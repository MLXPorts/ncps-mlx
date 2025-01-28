import mlx.core as mx
from ncps.mini_keras import ops
from ncps.mini_keras.backend.common import backend_utils

def vectorize(pyfunc, signature=None, excluded=None):
    """MLX implementation of vectorize"""
    def vmap_fn(func, in_axes):
        def wrapped(*args):
            # MLX doesn't have a direct vmap equivalent yet
            # This is a basic implementation that handles the most common case
            result = []
            for items in zip(*(arg for arg, ax in zip(args, in_axes) 
                             if ax is not None)):
                result.append(func(*items))
            return mx.stack(result)
        return wrapped

    return backend_utils.vectorize_impl(
        pyfunc, vmap_fn, excluded=excluded, signature=signature
    )
