"""Tests for ODE solvers."""

import pytest
import keras
import numpy as np
from keras import ops

from ncps.layers.ode_solvers import (
    euler_solve,
    rk4_solve,
    semi_implicit_solve,
    adaptive_heun_solve,
    solve_fixed_grid
)


def test_euler_solve():
    """Test Euler solver."""
    batch_size = 8
    state_size = 32
    
    # Test linear ODE: dy/dt = -y
    def f(t, y):
        return -y
    
    # Initial condition
    y0 = ops.ones((batch_size, state_size))
    
    # Solve with scalar dt
    dt = 0.1
    y1 = euler_solve(f, y0, dt)
    
    # Check shape
    assert y1.shape == y0.shape
    
    # Check solution (should be exp(-t))
    expected = np.exp(-dt)
    assert np.allclose(y1.numpy(), expected, rtol=1e-2)
    
    # Test with tensor dt
    dt = ops.ones((batch_size, 1)) * 0.1
    y1 = euler_solve(f, y0, dt)
    assert y1.shape == y0.shape


def test_rk4_solve():
    """Test RK4 solver."""
    batch_size = 8
    state_size = 32
    
    # Test harmonic oscillator: d²y/dt² = -y
    # Convert to system: dy/dt = v, dv/dt = -y
    def f(t, y):
        # y has shape [batch, 2*state_size] with position and velocity
        pos, vel = ops.split(y, 2, axis=-1)
        return ops.concatenate([vel, -pos], axis=-1)
    
    # Initial condition: y(0) = 1, v(0) = 0
    pos0 = ops.ones((batch_size, state_size))
    vel0 = ops.zeros((batch_size, state_size))
    y0 = ops.concatenate([pos0, vel0], axis=-1)
    
    # Solve with scalar dt
    dt = 0.1
    y1 = rk4_solve(f, y0, dt)
    
    # Check shape
    assert y1.shape == y0.shape
    
    # Check solution (should be cos(t) for position)
    pos1 = y1[:, :state_size]
    expected = np.cos(dt)
    assert np.allclose(pos1.numpy(), expected, rtol=1e-3)


def test_semi_implicit_solve():
    """Test semi-implicit solver."""
    batch_size = 8
    state_size = 32
    
    # Test stiff equation: dy/dt = -100y
    def f(t, y):
        return -100 * y
    
    # Initial condition
    y0 = ops.ones((batch_size, state_size))
    
    # Solve with scalar dt
    dt = 0.1
    y1 = semi_implicit_solve(f, y0, dt)
    
    # Check shape
    assert y1.shape == y0.shape
    
    # Solution should be more stable than explicit Euler
    y1_euler = euler_solve(f, y0, dt)
    
    # For stiff equation, semi-implicit should stay bounded
    assert ops.max(ops.abs(y1)) <= 1.0


def test_adaptive_heun_solve():
    """Test adaptive Heun solver."""
    batch_size = 8
    state_size = 32
    
    # Test nonlinear equation: dy/dt = y²
    def f(t, y):
        return y * y
    
    # Initial condition (small to avoid blow-up)
    y0 = ops.ones((batch_size, state_size)) * 0.1
    
    # Solve with different tolerances
    dt = 0.1
    y1 = adaptive_heun_solve(f, y0, dt, rtol=1e-3)
    y2 = adaptive_heun_solve(f, y0, dt, rtol=1e-6)
    
    # Check shapes
    assert y1.shape == y0.shape
    assert y2.shape == y0.shape
    
    # Higher tolerance should give more accurate result
    # but both should stay bounded for small initial condition
    assert ops.max(ops.abs(y1)) <= 0.2
    assert ops.max(ops.abs(y2)) <= 0.2


def test_solve_fixed_grid():
    """Test fixed grid solver."""
    batch_size = 8
    state_size = 32
    
    # Test linear ODE: dy/dt = -y
    def f(t, y):
        return -y
    
    # Initial condition
    y0 = ops.ones((batch_size, state_size))
    
    # Time points
    t = np.linspace(0, 1, 11)
    
    # Solve with different methods
    methods = ["euler", "rk4", "semi_implicit", "adaptive"]
    
    for method in methods:
        # Solve
        y = solve_fixed_grid(f, y0, t, method=method)
        
        # Check shape
        assert y.shape == (11, batch_size, state_size)
        
        # Check solution stays bounded
        assert ops.max(ops.abs(y)) <= 1.0
        
        # Check solution is decreasing (since dy/dt = -y)
        diffs = y[1:] - y[:-1]
        assert ops.all(diffs <= 0)
    
    # Test invalid method
    with pytest.raises(ValueError):
        solve_fixed_grid(f, y0, t, method="invalid")


def test_stiff_equation():
    """Test solvers with stiff equation."""
    batch_size = 8
    state_size = 32
    
    # Test stiff equation: dy/dt = -1000y
    def f(t, y):
        return -1000 * y
    
    # Initial condition
    y0 = ops.ones((batch_size, state_size))
    
    # Small time step
    dt = 0.001
    
    # Compare methods
    y1 = euler_solve(f, y0, dt)
    y2 = rk4_solve(f, y0, dt)
    y3 = semi_implicit_solve(f, y0, dt)
    y4 = adaptive_heun_solve(f, y0, dt)
    
    # All solutions should stay bounded
    assert ops.max(ops.abs(y1)) <= 1.0
    assert ops.max(ops.abs(y2)) <= 1.0
    assert ops.max(ops.abs(y3)) <= 1.0
    assert ops.max(ops.abs(y4)) <= 1.0


def test_system_equation():
    """Test solvers with system of equations."""
    batch_size = 8
    state_size = 32
    
    # Test Lotka-Volterra: dx/dt = ax - bxy, dy/dt = -cy + dxy
    def f(t, y):
        x, y = ops.split(y, 2, axis=-1)
        a, b = 2.0, 1.0
        c, d = 1.0, 1.0
        dx = a*x - b*x*y
        dy = -c*y + d*x*y
        return ops.concatenate([dx, dy], axis=-1)
    
    # Initial condition
    x0 = ops.ones((batch_size, state_size))
    y0 = ops.ones((batch_size, state_size))
    state0 = ops.concatenate([x0, y0], axis=-1)
    
    # Solve
    dt = 0.1
    state1 = rk4_solve(f, state0, dt)
    
    # Check shape
    assert state1.shape == state0.shape
    
    # Solution should preserve positivity
    x1, y1 = ops.split(state1, 2, axis=-1)
    assert ops.all(x1 >= 0)
    assert ops.all(y1 >= 0)