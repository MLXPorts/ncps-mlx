"""ODE solvers implemented in MLX."""

import mlx.core as mx


def euler_solve(f, y0, dt):
    """
    Euler method for ODE solving.
    
    Args:
        f (callable): Function representing dy/dt = f(y)
        y0 (mx.array): Initial state
        dt (float): Time step size
    
    Returns:
        mx.array: Solution at t + dt
    """
    return y0 + dt * f(None, y0)


def rk4_solve(f, y0, t0, dt):
    """
    4th order Runge-Kutta method for ODE solving.
    
    Args:
        f (callable): Function representing dy/dt = f(t, y)
        y0 (mx.array): Initial state
        t0 (float): Initial time
        dt (float): Time step size
    
    Returns:
        mx.array: Solution at t0 + dt
    """
    k1 = f(t0, y0)
    k2 = f(t0 + dt/2, y0 + dt*k1/2)
    k3 = f(t0 + dt/2, y0 + dt*k2/2)
    k4 = f(t0 + dt, y0 + dt*k3)
    
    return y0 + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def semi_implicit_solve(f, y0, dt):
    """
    Semi-implicit Euler method for ODE solving.
    
    Args:
        f (callable): Function representing dy/dt = f(y)
        y0 (mx.array): Initial state
        dt (float): Time step size
    
    Returns:
        mx.array: Solution at t + dt
    """
    # First get the derivative at the current point
    k1 = f(None, y0)
    
    # Then use this to estimate the next point
    y_pred = y0 + dt * k1
    
    # Get the derivative at the predicted point
    k2 = f(None, y_pred)
    
    # Average the derivatives and take a step
    return y0 + dt * (k1 + k2) / 2
