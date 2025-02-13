"""Enhanced Liquid Time-Constant Cell implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any, Callable

from .ltc_cell import LTCCell
from .ode_solvers import rk4_solve, euler_solve, semi_implicit_solve


class ELTCCell(LTCCell):
    """
    Enhanced Liquid Time-Constant Cell (ELTC) for MLX.

    This class extends the LTCCell implementation by adding:
    - Configurable solvers (e.g., RK4, Euler, Semi-Implicit)
    - Sparsity constraints for adjacency matrices
    - Flexible activation functions
    """

    def __init__(
        self,
        wiring,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        solver: str = "rk4",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        initialization_ranges: Optional[Dict[str, Any]] = None,
        forget_gate_bias: float = 1.0,
        sparsity: float = 0.5,
        activation: Callable = mx.tanh,
        **kwargs,
    ):
        """
        Initialize the EnhancedLTCCell.

        Args:
            wiring: Neural wiring pattern
            input_mapping: Input mapping type ("affine" or "linear")
            output_mapping: Output mapping type ("affine" or "linear")
            solver: Solver type for ODE solving ("rk4", "euler", "semi_implicit")
            ode_unfolds: Number of ODE unfolds per time step
            epsilon: Small constant to avoid division by zero
            initialization_ranges: Ranges for parameter initialization
            forget_gate_bias: Bias for the forget gate
            sparsity: Sparsity level for adjacency matrices
            activation: Activation function
            **kwargs: Additional arguments passed to the base LTCCell
        """
        super().__init__(
            wiring=wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            initialization_ranges=initialization_ranges,
            forget_gate_bias=forget_gate_bias,
            **kwargs,
        )
        self.solver = solver
        self.sparsity = sparsity
        self.activation = activation

    def build(self):
        """Initialize parameters."""
        super().build()
        self._apply_sparsity()

    def _apply_sparsity(self):
        """Apply sparsity constraints to the adjacency matrix."""
        if not hasattr(self, 'weight'):
            return  # Skip if params not yet initialized
            
        # Use bernoulli for sparsity mask
        mask = mx.random.bernoulli(1 - self.sparsity, self.weight.shape)
        self.weight = self.weight * mask

    def _sigmoid(self, v, mu, sigma):
        """Compute sigmoid activation."""
        return 1.0 / (1.0 + mx.exp(-(v - mu) / sigma))

    def ode_solver(self, f, y0, t0, dt):
        """
        Solve ODEs using the configured solver.

        Args:
            f: Function representing the ODE (dy/dt = f(y, t))
            y0: Initial state
            t0: Initial time
            dt: Time step size

        Returns:
            Updated state after one time step
        """
        if self.solver == "rk4":
            return rk4_solve(f, y0, t0, dt)
        elif self.solver == "euler":
            return euler_solve(f, y0, dt)
        elif self.solver == "semi_implicit":
            return semi_implicit_solve(f, y0, dt)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver}")

    def _ode_solver(self, inputs, state, elapsed_time):
        """
        Solve the ODEs with enhanced solvers.

        Args:
            inputs: Input tensor
            state: Current state tensor
            elapsed_time: Time elapsed since the last step

        Returns:
            Updated state tensor
        """
        v_pre = state

        # Precompute sensory neuron effects
        sensory_w_activation = self.sensory_weight * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )

        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = mx.sum(sensory_w_activation, axis=1)

        cm_t = self.cm / (elapsed_time / self._ode_unfolds)

        # ODE unfolds
        for _ in range(self._ode_unfolds):
            def f(_, v_pre):
                w_activation = self.weight * self._sigmoid(v_pre, self.mu, self.sigma)
                rev_activation = w_activation * self.erev
                
                w_numerator = mx.sum(rev_activation, axis=1) + w_numerator_sensory
                w_denominator = mx.sum(w_activation, axis=1) + w_denominator_sensory

                numerator = cm_t * v_pre + self.gleak * self.vleak + w_numerator
                denominator = cm_t + self.gleak + w_denominator

                return numerator / (denominator + self.epsilon)

            v_pre = self.ode_solver(f, v_pre, 0, 1.0 / self._ode_unfolds)

        return v_pre

    def __call__(self, inputs, state=None, time=1.0):
        """
        Process one time step.

        Args:
            inputs: Input tensor
            state: Optional initial state
            time: Time step size

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = inputs.shape[0]
        if state is None:
            state = mx.zeros((batch_size, self.units))

        new_state = self._ode_solver(inputs, state, time)
        output = self.activation(new_state)

        if self.output_dim != self.units:
            output = output[:, :self.output_dim]

        return output, new_state

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "solver": self.solver,
            "sparsity": self.sparsity,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        base_config = LTCCell.from_config(config)
        return cls(
            **base_config,
            solver=config["solver"],
            sparsity=config["sparsity"],
            activation=config["activation"],
        )
