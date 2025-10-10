"""Simple ODE solvers implemented with MLX operations."""

from __future__ import annotations

import mlx.core as mx


def euler_solve(f, y0, dt):
    """Explicit Euler step."""
    return y0 + dt * f(None, y0)


def rk4_solve(f, y0, t0, dt):
    """Fourth-order Runge–Kutta integration."""
    k1 = f(t0, y0)
    k2 = f(t0 + dt / 2, y0 + dt * k1 / 2)
    k3 = f(t0 + dt / 2, y0 + dt * k2 / 2)
    k4 = f(t0 + dt, y0 + dt * k3)
    return y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def semi_implicit_solve(f, y0, dt):
    """Semi-implicit Euler step (Heun's method)."""
    k1 = f(None, y0)
    y_pred = y0 + dt * k1
    k2 = f(None, y_pred)
    return y0 + dt * (k1 + k2) / 2
