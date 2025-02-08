import numpy as np

def rk4_solve(f, y0, t0, dt):
    """RK4 solver using NumPy ops exclusively."""
    k1 = f(t0, y0)
    
    t_half = t0 + dt * 0.5
    dt_k1_half = dt * k1 * 0.5
    y_half = y0 + dt_k1_half
    k2 = f(t_half, y_half)
    
    dt_k2_half = dt * k2 * 0.5
    y_half_2 = y0 + dt_k2_half
    k3 = f(t_half, y_half_2)
    
    t_full = t0 + dt
    dt_k3 = dt * k3
    y_full = y0 + dt_k3
    k4 = f(t_full, y_full)
    
    k_sum = k1 + 2.0 * k2 + 2.0 * k3 + k4
    return y0 + dt * k_sum / 6.0

def rk45_solve(f, y0, t0, dt):
    """RK45 solver using NumPy ops exclusively."""
    # Butcher tableau coefficients as NumPy constants
    a2 = 1.0 / 4.0
    a3 = 3.0 / 8.0
    a4 = 12.0 / 13.0
    a5 = 1.0
    a6 = 1.0 / 2.0
    
    b21 = 1.0 / 4.0
    b31 = 3.0 / 32.0
    b32 = 9.0 / 32.0
    b41 = 1932.0 / 2197.0
    b42 = -7200.0 / 2197.0
    b43 = 7296.0 / 2197.0
    b51 = 439.0 / 216.0
    b52 = -8.0
    b53 = 3680.0 / 513.0
    b54 = -845.0 / 4104.0
    b61 = -8.0 / 27.0
    b62 = 2.0
    b63 = -3544.0 / 2565.0
    b64 = 1859.0 / 4104.0
    b65 = -11.0 / 40.0
    
    k1 = f(t0, y0)
    
    t2 = t0 + dt * a2
    y2 = y0 + dt * k1 * b21
    k2 = f(t2, y2)
    
    t3 = t0 + dt * a3
    y3 = y0 + dt * (k1 * b31 + k2 * b32)
    k3 = f(t3, y3)
    
    t4 = t0 + dt * a4
    y4 = y0 + dt * (k1 * b41 + k2 * b42 + k3 * b43)
    k4 = f(t4, y4)
    
    t5 = t0 + dt * a5
    y5 = y0 + dt * (k1 * b51 + k2 * b52 + k3 * b53 + k4 * b54)
    k5 = f(t5, y5)
    
    t6 = t0 + dt * a6
    y6 = y0 + dt * (k1 * b61 + k2 * b62 + k3 * b63 + k4 * b64 + k5 * b65)
    k6 = f(t6, y6)
    
    # 5th order solution coefficients
    c1 = 16.0 / 135.0
    c3 = 6656.0 / 12825.0
    c4 = 28561.0 / 56430.0
    c5 = -9.0 / 50.0
    c6 = 2.0 / 55.0
    
    return y0 + dt * (k1 * c1 + k3 * c3 + k4 * c4 + k5 * c5 + k6 * c6)

def euler_solve(f, y0, t0, dt):
    """Euler solver using NumPy ops exclusively."""
    return y0 + dt * f(t0, y0)

def heun_solve(f, y0, t0, dt):
    """Heun solver using NumPy ops exclusively."""
    k1 = f(t0, y0)
    t_next = t0 + dt
    y_next = y0 + dt * k1
    k2 = f(t_next, y_next)
    k_avg = (k1 + k2) * 0.5
    return y0 + dt * k_avg

def midpoint_solve(f, y0, t0, dt):
    """Midpoint solver using NumPy ops exclusively."""
    k1 = f(t0, y0)
    t_mid = t0 + dt * 0.5
    y_mid = y0 + dt * k1 * 0.5
    k2 = f(t_mid, y_mid)
    return y0 + dt * k2

def ralston_solve(f, y0, t0, dt):
    """Ralston solver using NumPy ops exclusively."""
    k1 = f(t0, y0)
    t2 = t0 + dt * 2.0 / 3.0
    y2 = y0 + dt * k1 * 2.0 / 3.0
    k2 = f(t2, y2)
    k_combo = k1 * 0.25 + k2 * 0.75
    return y0 + dt * k_combo

def semi_implicit_solve(f, y0, t0, dt):
    """Semi-implicit solver using NumPy ops exclusively."""
    f_eval = f(t0, y0)
    diff = f_eval - y0
    return y0 + dt * diff
