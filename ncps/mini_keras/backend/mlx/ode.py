import mlx.core as mx

def rk4_solve(f, y0, t0, dt):
    """RK4 solver using MLX ops exclusively."""
    k1 = f(t0, y0)
    
    t_half = mx.add(t0, mx.multiply(dt, 0.5))
    dt_k1_half = mx.multiply(dt, mx.multiply(k1, 0.5))
    y_half = mx.add(y0, dt_k1_half)
    k2 = f(t_half, y_half)
    
    dt_k2_half = mx.multiply(dt, mx.multiply(k2, 0.5))
    y_half_2 = mx.add(y0, dt_k2_half)
    k3 = f(t_half, y_half_2)
    
    t_full = mx.add(t0, dt)
    dt_k3 = mx.multiply(dt, k3)
    y_full = mx.add(y0, dt_k3)
    k4 = f(t_full, y_full)
    
    k_sum = mx.add(
        k1,
        mx.add(
            mx.multiply(k2, 2.0),
            mx.add(
                mx.multiply(k3, 2.0),
                k4
            )
        )
    )
    return mx.add(y0, mx.multiply(dt, mx.multiply(k_sum, 1.0/6.0)))

def rk45_solve(f, y0, t0, dt):
    """RK45 solver using MLX ops exclusively."""
    # Butcher tableau coefficients as MLX constants
    a2 = mx.array(1.0/4.0)
    a3 = mx.array(3.0/8.0)
    a4 = mx.array(12.0/13.0)
    a5 = mx.array(1.0)
    a6 = mx.array(1.0/2.0)
    
    b21 = mx.array(1.0/4.0)
    b31 = mx.array(3.0/32.0)
    b32 = mx.array(9.0/32.0)
    b41 = mx.array(1932.0/2197.0)
    b42 = mx.array(-7200.0/2197.0)
    b43 = mx.array(7296.0/2197.0)
    b51 = mx.array(439.0/216.0)
    b52 = mx.array(-8.0)
    b53 = mx.array(3680.0/513.0)
    b54 = mx.array(-845.0/4104.0)
    b61 = mx.array(-8.0/27.0)
    b62 = mx.array(2.0)
    b63 = mx.array(-3544.0/2565.0)
    b64 = mx.array(1859.0/4104.0)
    b65 = mx.array(-11.0/40.0)
    
    k1 = f(t0, y0)
    
    t2 = mx.add(t0, mx.multiply(dt, a2))
    y2 = mx.add(y0, mx.multiply(dt, mx.multiply(k1, b21)))
    k2 = f(t2, y2)
    
    t3 = mx.add(t0, mx.multiply(dt, a3))
    y3 = mx.add(
        y0,
        mx.multiply(dt, 
            mx.add(
                mx.multiply(k1, b31),
                mx.multiply(k2, b32)
            )
        )
    )
    k3 = f(t3, y3)
    
    t4 = mx.add(t0, mx.multiply(dt, a4))
    y4 = mx.add(
        y0,
        mx.multiply(dt,
            mx.add(
                mx.add(
                    mx.multiply(k1, b41),
                    mx.multiply(k2, b42)
                ),
                mx.multiply(k3, b43)
            )
        )
    )
    k4 = f(t4, y4)
    
    t5 = mx.add(t0, mx.multiply(dt, a5))
    y5 = mx.add(
        y0,
        mx.multiply(dt,
            mx.add(
                mx.add(
                    mx.add(
                        mx.multiply(k1, b51),
                        mx.multiply(k2, b52)
                    ),
                    mx.multiply(k3, b53)
                ),
                mx.multiply(k4, b54)
            )
        )
    )
    k5 = f(t5, y5)
    
    t6 = mx.add(t0, mx.multiply(dt, a6))
    y6 = mx.add(
        y0,
        mx.multiply(dt,
            mx.add(
                mx.add(
                    mx.add(
                        mx.add(
                            mx.multiply(k1, b61),
                            mx.multiply(k2, b62)
                        ),
                        mx.multiply(k3, b63)
                    ),
                    mx.multiply(k4, b64)
                ),
                mx.multiply(k5, b65)
            )
        )
    )
    k6 = f(t6, y6)
    
    # 5th order solution coefficients
    c1 = mx.array(16.0/135.0)
    c3 = mx.array(6656.0/12825.0)
    c4 = mx.array(28561.0/56430.0)
    c5 = mx.array(-9.0/50.0)
    c6 = mx.array(2.0/55.0)
    
    return mx.add(
        y0,
        mx.multiply(dt,
            mx.add(
                mx.add(
                    mx.add(
                        mx.add(
                            mx.multiply(k1, c1),
                            mx.multiply(k3, c3)
                        ),
                        mx.multiply(k4, c4)
                    ),
                    mx.multiply(k5, c5)
                ),
                mx.multiply(k6, c6)
            )
        )
    )

def euler_solve(f, y0, t0, dt):
    """Euler solver using MLX ops exclusively."""
    return mx.add(y0, mx.multiply(dt, f(t0, y0)))

def heun_solve(f, y0, t0, dt):
    """Heun solver using MLX ops exclusively."""
    k1 = f(t0, y0)
    t_next = mx.add(t0, dt)
    y_next = mx.add(y0, mx.multiply(dt, k1))
    k2 = f(t_next, y_next)
    k_avg = mx.multiply(mx.add(k1, k2), 0.5)
    return mx.add(y0, mx.multiply(dt, k_avg))

def midpoint_solve(f, y0, t0, dt):
    """Midpoint solver using MLX ops exclusively."""
    k1 = f(t0, y0)
    t_mid = mx.add(t0, mx.multiply(dt, 0.5))
    y_mid = mx.add(y0, mx.multiply(mx.multiply(dt, k1), 0.5))
    k2 = f(t_mid, y_mid)
    return mx.add(y0, mx.multiply(dt, k2))

def ralston_solve(f, y0, t0, dt):
    """Ralston solver using MLX ops exclusively."""
    k1 = f(t0, y0)
    t2 = mx.add(t0, mx.multiply(dt, 2.0/3.0))
    y2 = mx.add(y0, mx.multiply(mx.multiply(dt, k1), 2.0/3.0))
    k2 = f(t2, y2)
    k_combo = mx.add(
        mx.multiply(k1, 0.25),
        mx.multiply(k2, 0.75)
    )
    return mx.add(y0, mx.multiply(dt, k_combo))

def semi_implicit_solve(f, y0, t0, dt):
    """Semi-implicit solver using MLX ops exclusively."""
    f_eval = f(t0, y0)
    diff = mx.subtract(f_eval, y0)
    return mx.add(y0, mx.multiply(dt, diff))
