import math
import mlx.core as mx
from mlx.core import matmul
from ncps.mini_keras.backend.mlx.hpc import HPC16x8  # For non-square matrix support

def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

def _custom_qr(matrix_high, matrix_low):
    """MLX-specific QR decomposition with increased numerical stability."""
    rows, cols = matrix_high.shape
    
    # Use HPC implementation for non-square matrices
    if rows != cols:
        matrix_hpc = HPC16x8.from_array(matrix_high)  # Convert to HPC format
        q_hpc, r_hpc = matrix_hpc.qr()  # HPC QR decomposition 
        return q_hpc.to_float32(), r_hpc.to_float32(), None
    
    # Square matrix case - use existing implementation
    q_high = mx.zeros((rows, cols), dtype=mx.float32)
    r_high = mx.zeros((cols, cols), dtype=mx.float32)
    r_low  = mx.zeros((cols, cols), dtype=mx.float32)

    for i in range(cols):
        v_high, v_low = matrix_high[:, i], matrix_low[:, i]

        for j in range(i):
            r_high[j, i] = matmul(q_high[:, j].reshape(1, -1), v_high.reshape(-1, 1)).item()
            r_low[j, i]  = (
                matmul(q_high[:, j].reshape(1, -1), v_low.reshape(-1, 1))
                + matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_high.reshape(-1, 1))
                + matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_low.reshape(-1, 1))
            ).item()

            proj_high = matmul(q_high[:, j].reshape(-1, 1), mx.array(r_high[j, i]).reshape(1, 1))
            proj_low  = matmul(q_high[:, j].reshape(-1, 1), mx.array(r_low[j, i]).reshape(1, 1))

            v_high, v_low = _add_double_single(v_high, v_low, -proj_high[:, 0], -proj_low[:, 0])
            matrix_high[:, i], matrix_low[:, i] = v_high, v_low

        norm_high = mx.linalg.norm(v_high)
        if norm_high < 1e-10:
            raise ValueError(f"Column norm too small (col={i}). Check initialization.")

        q_high[:, i] = (v_high / norm_high).astype(mx.float32)

    return q_high, r_high, r_low

def orthogonal(shape, gain=1.0):
    """MLX-specific orthogonal matrix initialization with improved numerical stability.
    Uses HPC implementation for non-square matrices to handle MLX limitations."""
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    rows, cols = shape[0], math.prod(shape[1:])
    size = max(rows, cols)

    matrix_high = mx.random.normal(
        shape=(size, size),
        dtype=mx.float32,
        loc=0.0,
        scale=1.0
    )
    
    if rows != cols:
        # Use HPC path for non-square
        matrix_hpc = HPC16x8.from_array(matrix_high)
        q_high = matrix_hpc.qr()[0].to_float32()
    else:
        # Square matrix - use standard path
        matrix_low = mx.random.normal(
            shape=(size, size), 
            dtype=mx.float32,
            loc=0, 
            scale=1e-7
        )
        q_high, _, _ = _custom_qr(matrix_high, matrix_low)
    
    q_high = q_high[:rows, :cols]
    return gain * q_high.reshape(shape)
