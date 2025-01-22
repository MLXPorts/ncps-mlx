import math
import mlx.core as mx
from mlx.core import matmul, swapaxes, eye, allclose


def add_double_single(a_high, a_low, b_high, b_low):
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

def custom_qr(matrix_high, matrix_low):
    rows, cols = matrix_high.shape
    q_high = mx.zeros((rows, cols), dtype=mx.float32)
    r_high = mx.zeros((cols, cols), dtype=mx.float32)
    r_low  = mx.zeros((cols, cols), dtype=mx.float32)

    for i in range(cols):
        v_high, v_low = matrix_high[:, i], matrix_low[:, i]

        for j in range(i):
            # Debug: print vectors before projection
            # print(f"q_high[:, {j}] = {q_high[:, j]}")
            # print(f"v_high       = {v_high}")

            r_high[j, i] = matmul(q_high[:, j].reshape(1, -1), v_high.reshape(-1, 1)).item()
            r_low[j, i]  = (
                matmul(q_high[:, j].reshape(1, -1), v_low.reshape(-1, 1))
                + matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_high.reshape(-1, 1))
                + matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_low.reshape(-1, 1))
            ).item()

            proj_high = matmul(q_high[:, j].reshape(-1, 1), mx.array(r_high[j, i]).reshape(1, 1))
            proj_low  = matmul(q_high[:, j].reshape(-1, 1), mx.array(r_low[j, i]).reshape(1, 1))

            v_high, v_low = add_double_single(v_high, v_low, -proj_high[:, 0], -proj_low[:, 0])

            # If you see v_high ~0, consider reinitializing to avoid degenerate columns
            # if mx.linalg.norm(v_high) < 1e-10:
            #     matrix_high[:, i] = mx.random.normal((rows,), 0.0, 1.0, dtype=mx.float32)
            #     matrix_low[:, i]  = mx.random.normal((rows,), 0.0, 1e-7, dtype=mx.float32)
            #     v_high, v_low = matrix_high[:, i], matrix_low[:, i]
            #     print(f"Reinitialized column {i}")

            matrix_high[:, i], matrix_low[:, i] = v_high, v_low

        norm_high = mx.linalg.norm(v_high)
        if norm_high < 1e-10:
            raise ValueError(f"Column norm too small (col={i}). Check initialization.")

        q_high[:, i] = (v_high / norm_high).astype(mx.float32)

    return q_high, r_high, r_low

def orthogonal(shape, gain=1.0):
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
    
    matrix_low  = mx.random.normal(
        shape=(size, size), 
        dtype=mx.float32,
        loc = 0, 
        scale = 1e-7
    )

    q_high, r_h, r_l = custom_qr(matrix_high, matrix_low)
    # Debug: check r_h for non-zero entries
    # print("r_high = ", r_h)
    # print("r_low  = ", r_l)

    q_high = q_high[:rows, :cols]
    return gain * q_high.reshape(shape)

if __name__ == "__main__":
    shape = (4, 6)
    tensor = orthogonal(shape)
    q_t_q = matmul(tensor, swapaxes(tensor, 0, 1))
    assert allclose(q_t_q, eye(min(shape)), atol=1e-5)
    print("Orthogonal initializer passed.")