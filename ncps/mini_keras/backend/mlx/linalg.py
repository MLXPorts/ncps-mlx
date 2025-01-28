import mlx.core as mx
from ncps.mini_keras.backend import standardize_dtype
from ncps.mini_keras.backend.common import dtypes
from ncps.mini_keras.backend.mlx.core import convert_to_tensor

def cholesky(a):
    return mx.linalg.cholesky(a)

def det(a):
    return mx.linalg.det(a)

def eig(a):
    return mx.linalg.eig(a)

def eigh(a):
    return mx.linalg.eigh(a)

def inv(a):
    return mx.linalg.inv(a)

def lu_factor(a):
    # Note: MLX currently doesn't have direct LU factorization
    # This is a placeholder that needs implementation
    raise NotImplementedError(
        "LU factorization not yet implemented in MLX backend"
    )

def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    return mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims).astype(
        dtype
    )

def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return mx.linalg.qr(x)  # Note: MLX qr doesn't support mode parameter currently

def solve(a, b):
    return mx.linalg.solve(a, b)

def solve_triangular(a, b, lower=False):
    # Note: MLX doesn't have a direct triangular solver
    # This is a placeholder that needs implementation
    raise NotImplementedError(
        "Triangular solve not yet implemented in MLX backend"
    )

def svd(x, full_matrices=True, compute_uv=True):
    return mx.linalg.svd(x)  # Note: MLX svd doesn't support all numpy parameters currently

def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # Note: MLX doesn't have a direct least squares solver
    # This uses the normal equation method as a fallback
    return solve(mx.matmul(mx.transpose(a), a), mx.matmul(mx.transpose(a), b))
