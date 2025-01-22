from ncps.mini_keras import backend
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.backend import KerasTensor
from ncps.mini_keras.backend import any_symbolic_tensors
from ncps.mini_keras.ops.operation import Operation

class RK4Solve(Operation):
    def call(self, f, y0, t0, dt):
        k1 = f(t0, y0)
        k2 = f(t0 + dt / 2, y0 + dt * k1 / 2)
        k3 = f(t0 + dt / 2, y0 + dt * k2 / 2)
        k4 = f(t0 + dt, y0 + dt * k3)
        return y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

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
        """Runge-Kutta-Fehlberg (RKF45) method with adaptive step size."""
        # Butcher tableau coefficients
        a2 = 1/4; a3 = 3/8; a4 = 12/13; a5 = 1; a6 = 1/2
        b21 = 1/4
        b31 = 3/32; b32 = 9/32
        b41 = 1932/2197; b42 = -7200/2197; b43 = 7296/2197
        b51 = 439/216; b52 = -8; b53 = 3680/513; b54 = -845/4104
        b61 = -8/27; b62 = 2; b63 = -3544/2565; b64 = 1859/4104; b65 = -11/40
        
        k1 = f(t0, y0)
        k2 = f(t0 + a2*dt, y0 + dt*(b21*k1))
        k3 = f(t0 + a3*dt, y0 + dt*(b31*k1 + b32*k2))
        k4 = f(t0 + a4*dt, y0 + dt*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t0 + a5*dt, y0 + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t0 + a6*dt, y0 + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        
        # 5th order solution
        return y0 + dt * (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)

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
        return y0 + dt * f(t0, y0)

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
        """Heun's method (improved Euler)."""
        k1 = f(t0, y0)
        k2 = f(t0 + dt, y0 + dt * k1)
        return y0 + dt * (k1 + k2) / 2

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
        """Midpoint method."""
        k1 = f(t0, y0)
        k2 = f(t0 + dt/2, y0 + dt * k1/2)
        return y0 + dt * k2

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
        """Ralston's method (optimal 2nd order)."""
        k1 = f(t0, y0)
        k2 = f(t0 + 2*dt/3, y0 + 2*dt * k1/3)
        return y0 + dt * (k1/4 + 3*k2/4)

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
        return y0 + dt * (f(t0, y0) - y0)

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
