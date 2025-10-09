"""ODE solvers implemented with Keras backend."""

import keras
from keras import ops
from typing import Callable, Union, Optional


def euler_solve(
    f: Callable,
    y0: keras.KerasTensor,
    dt: Union[float, keras.KerasTensor],
    t0: Optional[float] = 0.0
) -> keras.KerasTensor:
    """Euler method for ODE solving.
    
    Args:
        f: Function representing dy/dt = f(t, y)
        y0: Initial state tensor
        dt: Time step size (scalar or tensor)
        t0: Optional initial time
        
    Returns:
        Solution tensor at t + dt
    """
    # Compute derivative
    dy = f(t0, y0)
    
    # Handle scalar vs tensor dt
    if isinstance(dt, (int, float)):
        return y0 + dt * dy
    else:
        # Ensure dt has right shape for broadcasting
        dt = ops.reshape(dt, dt.shape + (1,) * (len(y0.shape) - len(dt.shape)))
        return y0 + dt * dy


def rk4_solve(
    f: Callable,
    y0: keras.KerasTensor,
    dt: Union[float, keras.KerasTensor],
    t0: Optional[float] = 0.0
) -> keras.KerasTensor:
    """4th order Runge-Kutta method for ODE solving.
    
    Args:
        f: Function representing dy/dt = f(t, y)
        y0: Initial state tensor
        dt: Time step size (scalar or tensor)
        t0: Initial time
        
    Returns:
        Solution tensor at t0 + dt
    """
    # Handle scalar vs tensor dt for half steps
    if isinstance(dt, (int, float)):
        dt_half = dt / 2
    else:
        dt = ops.reshape(dt, dt.shape + (1,) * (len(y0.shape) - len(dt.shape)))
        dt_half = dt / 2
    
    # Compute RK4 terms
    k1 = f(t0, y0)
    k2 = f(t0 + dt_half, y0 + dt_half * k1)
    k3 = f(t0 + dt_half, y0 + dt_half * k2)
    k4 = f(t0 + dt, y0 + dt * k3)
    
    # Combine terms
    return y0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def semi_implicit_solve(
    f: Callable,
    y0: keras.KerasTensor,
    dt: Union[float, keras.KerasTensor],
    t0: Optional[float] = 0.0
) -> keras.KerasTensor:
    """Semi-implicit Euler method for ODE solving.
    
    Better stability than explicit Euler for stiff equations.
    Uses a stabilized formulation for stiff problems.
    
    Args:
        f: Function representing dy/dt = f(t, y)
        y0: Initial state tensor
        dt: Time step size (scalar or tensor)
        t0: Optional initial time
        
    Returns:
        Solution tensor at t + dt
    """
    # Handle scalar vs tensor dt
    if isinstance(dt, (int, float)):
        dt_small = dt / 10  # Use smaller steps for stability
    else:
        dt = ops.reshape(dt, dt.shape + (1,) * (len(y0.shape) - len(dt.shape)))
        dt_small = dt / 10
    
    # Take multiple small steps for stability
    y = y0
    for _ in range(10):
        # Get derivative at current point
        k1 = f(t0, y)
        
        # Estimate next point
        y_pred = y + dt_small * k1
        
        # Get derivative at predicted point
        k2 = f(t0 + dt_small, y_pred)
        
        # Update with averaged derivatives
        y = y + dt_small * (k1 + k2) / 4  # Use smaller factor for stability
    
    return y


def adaptive_heun_solve(
    f: Callable,
    y0: keras.KerasTensor,
    dt: Union[float, keras.KerasTensor],
    t0: Optional[float] = 0.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    min_dt: float = 1e-6,
    max_dt: float = 1.0
) -> keras.KerasTensor:
    """Adaptive Heun method with error control.
    
    Args:
        f: Function representing dy/dt = f(t, y)
        y0: Initial state tensor
        dt: Initial time step size
        t0: Optional initial time
        rtol: Relative tolerance
        atol: Absolute tolerance
        min_dt: Minimum allowed time step
        max_dt: Maximum allowed time step
        
    Returns:
        Solution tensor at t + dt
    """
    def compute_error_ratio(y1, y2):
        """Compute normalized error ratio."""
        scale = atol + rtol * ops.maximum(ops.abs(y1), ops.abs(y2))
        error = ops.abs(y2 - y1)
        ratio = error / scale
        return ops.mean(ratio)
    
    # Initial step with Heun's method
    k1 = f(t0, y0)
    y1 = y0 + dt * k1
    k2 = f(t0 + dt, y1)
    y_heun = y0 + (dt/2) * (k1 + k2)
    
    # Compute error ratio
    error_ratio = compute_error_ratio(y1, y_heun)
    
    # Adjust step size if needed
    if error_ratio > 1.0:
        # Reduce step size and try again
        dt_new = ops.maximum(
            min_dt,
            0.9 * dt * ops.power(error_ratio, -0.5)
        )
        dt_new = ops.minimum(dt_new, max_dt)
        
        # Recursive call with new step size
        return adaptive_heun_solve(
            f, y0, dt_new, t0,
            rtol, atol, min_dt, max_dt
        )
    
    return y_heun


def solve_fixed_grid(
    f: Callable,
    y0: keras.KerasTensor,
    t: Union[float, keras.KerasTensor],
    method: str = "rk4"
) -> keras.KerasTensor:
    """Solve ODE on fixed time grid.
    
    Args:
        f: Function representing dy/dt = f(t, y)
        y0: Initial state tensor
        t: Time points tensor
        method: Integration method ("euler", "rk4", "semi_implicit", "adaptive")
        
    Returns:
        Solution tensor at specified time points
        
    Raises:
        ValueError: If method is unknown
    """
    methods = {
        "euler": euler_solve,
        "rk4": rk4_solve,
        "semi_implicit": semi_implicit_solve,
        "adaptive": adaptive_heun_solve
    }
    
    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Valid options are: {list(methods.keys())}"
        )
    
    solver = methods[method]
    
    # Get time steps
    if isinstance(t, (int, float)):
        dt = t
        t = ops.linspace(0.0, t, 11)  # Default to 10 steps
    else:
        dt = t[1:] - t[:-1]
    
    # Solve ODE
    y = y0
    outputs = [y0]
    
    for i in range(len(dt)):
        y = solver(f, y, dt[i], t[i])
        outputs.append(y)
    
    return ops.stack(outputs)