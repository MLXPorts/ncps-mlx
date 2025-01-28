from ncps.mini_keras import backend
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.backend import KerasTensor
from ncps.mini_keras.backend import any_symbolic_tensors
from ncps.mini_keras.ops.operation import Operation

class RK4Solve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.rk4_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.rk4_solve", "ncps.mini_keras.ops.ode.rk4_solve"])
def rk4_solve(f, y0, t0, dt):
    """Runge-Kutta 4th order solver."""
    if any_symbolic_tensors((y0,)):
        return RK4Solve().symbolic_call(f, y0, t0, dt)
    return backend.ode.rk4_solve(f, y0, t0, dt)


class RK45Solve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.rk45_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.rk45_solve", "ncps.mini_keras.ops.ode.rk45_solve"])
def rk45_solve(f, y0, t0, dt):
    """Runge-Kutta-Fehlberg (RKF45) solver with adaptive step size."""
    if any_symbolic_tensors((y0,)):
        return RK45Solve().symbolic_call(f, y0, t0, dt)
    return backend.ode.rk45_solve(f, y0, t0, dt)


class EulerSolve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.euler_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.euler_solve", "ncps.mini_keras.ops.ode.euler_solve"])
def euler_solve(f, y0, t0, dt):
    """Forward Euler method."""
    if any_symbolic_tensors((y0,)):
        return EulerSolve().symbolic_call(f, y0, t0, dt)
    return backend.ode.euler_solve(f, y0, t0, dt)


class HeunSolve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.heun_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.heun_solve", "ncps.mini_keras.ops.ode.heun_solve"])
def heun_solve(f, y0, t0, dt):
    """Heun's method (improved Euler)."""
    if any_symbolic_tensors((y0,)):
        return HeunSolve().symbolic_call(f, y0, t0, dt)
    return backend.ode.heun_solve(f, y0, t0, dt)


class MidpointSolve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.midpoint_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.midpoint_solve", "ncps.mini_keras.ops.ode.midpoint_solve"])
def midpoint_solve(f, y0, t0, dt):
    """Midpoint method."""
    if any_symbolic_tensors((y0,)):
        return MidpointSolve().symbolic_call(f, y0, t0, dt)
    return backend.ode.midpoint_solve(f, y0, t0, dt)


class RalstonSolve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.ralston_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.ralston_solve", "ncps.mini_keras.ops.ode.ralston_solve"])
def ralston_solve(f, y0, t0, dt):
    """Ralston's method (optimal 2nd order)."""
    if any_symbolic_tensors((y0,)):
        return RalstonSolve().symbolic_call(f, y0, t0, dt)
    return backend.ode.ralston_solve(f, y0, t0, dt)


class SemiImplicitSolve(Operation):
    def call(self, f, y0, t0, dt):
        return backend.ode.semi_implicit_solve(f, y0, t0, dt)

    def compute_output_spec(self, f, y0, t0, dt):
        return KerasTensor(y0.shape, dtype=y0.dtype)


@keras_mini_export(["ncps.mini_keras.ops.semi_implicit_solve", "ncps.mini_keras.ops.ode.semi_implicit_solve"])
def semi_implicit_solve(f, y0, t0, dt):
    """Semi-implicit method."""
    if any_symbolic_tensors((y0,)):
        return SemiImplicitSolve().symbolic_call(f, y0, t0, dt)
    return backend.ode.semi_implicit_solve(f, y0, t0, dt)


SOLVERS = {
    "euler": euler_solve,
    "heun": heun_solve,
    "midpoint": midpoint_solve,
    "ralston": ralston_solve,
    "rk4": rk4_solve,
    "rk45": rk45_solve,
    "semi_implicit": semi_implicit_solve,
}
