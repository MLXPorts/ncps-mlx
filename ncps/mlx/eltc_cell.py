# Copyright 2025 Sydney Renee
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Credits: This implementation builds upon the work of Mathias Lechner and Ramin Hasani,
# incorporating improvements for flexibility, solver options, and sparsity constraints.

import mlx.core as mx
from .ltc_cell import LTCCell
import ncps

@ncps.mini_keras.utils.register_keras_serializable(package="ncps", name="EnhancedLTCCell")
class EnhancedLTCCell(LTCCell):
    """
    Enhanced Liquid Time-Constant Cell (LTC-SE) for MLX.

    This class extends the LTCCell implementation by adding:
    - Configurable solvers (e.g., RK4, Euler, Semi-Implicit).
    - Sparsity constraints for adjacency matrices.
    - Flexible activation functions.
    - Modular serialization support.

    Attributes:
        solver (str): Type of solver to use for ODEs ("rk4", "euler", "semi_implicit").
        sparsity (float): Sparsity level for adjacency matrices (default: 0.5).
        activation (callable): Activation function to use (default: mx.tanh).
    """

    def __init__(
        self,
        wiring,
        input_mapping="affine",
        output_mapping="affine",
        solver="rk4",
        ode_unfolds=6,
        epsilon=1e-8,
        initialization_ranges=None,
        forget_gate_bias=1.0,
        sparsity=0.5,
        activation=mx.tanh,
        **kwargs,
    ):
        """
        Initialize the EnhancedLTCCell.

        Args:
            wiring (ncps.wirings.Wiring): Wiring configuration for the LTC cell.
            input_mapping (str): Input mapping type ("affine" or "linear"). Default: "affine".
            output_mapping (str): Output mapping type ("affine" or "linear"). Default: "affine".
            solver (str): Solver type for ODE solving ("rk4", "euler", "semi_implicit"). Default: "rk4".
            ode_unfolds (int): Number of ODE unfolds per time step. Default: 6.
            epsilon (float): Small constant to avoid division by zero. Default: 1e-8.
            initialization_ranges (dict): Ranges for parameter initialization. Default: None.
            forget_gate_bias (float): Bias for the forget gate. Default: 1.0.
            sparsity (float): Sparsity level for adjacency matrices. Default: 0.5.
            activation (callable): Activation function. Default: mx.tanh.
            kwargs: Additional arguments passed to the base LTCCell.
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

        # Apply sparsity to the adjacency matrix
        self._apply_sparsity()

    def _apply_sparsity(self):
        """
        Apply sparsity constraints to the adjacency matrix.
        """
        if not hasattr(self, '_params'):
            return  # Skip if params not yet initialized
            
        # Use bernoulli instead of choice
        mask = mx.random.bernoulli(1 - self.sparsity, self._params["w"].shape)
        self._params["w"] = self._params["w"] * mask

    def ode_solver(self, f, y0, t0, dt):
        """
        Solve ODEs using the configured solver.

        Args:
            f (callable): Function representing the ODE (dy/dt = f(y, t)).
            y0 (mx.array): Initial state.
            t0 (float): Initial time.
            dt (float): Time step size.

        Returns:
            mx.array: Updated state after one time step.
        """
        if self.solver == "rk4":
            return self._rk4_solver(f, y0, t0, dt)
        elif self.solver == "euler":
            return y0 + dt * f(t0, y0)
        elif self.solver == "semi_implicit":
            return self._semi_implicit_solver(f, y0, dt)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver}")

    def _rk4_solver(self, f, y0, t0, dt):
        """
        Runge-Kutta 4th order solver.
        """
        k1 = f(t0, y0)
        k2 = f(t0 + dt / 2, y0 + dt * k1 / 2)
        k3 = f(t0 + dt / 2, y0 + dt * k2 / 2)
        k4 = f(t0 + dt, y0 + dt * k3)
        return y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _semi_implicit_solver(self, f, y0, dt):
        """
        Semi-Implicit solver.
        """
        return y0 + dt * (f(0, y0) - y0)

    def _ode_solver(self, inputs, state, elapsed_time):
        """
        Solve the ODEs with enhanced solvers.

        Args:
            inputs (mx.array): Input tensor.
            state (mx.array): Current state tensor.
            elapsed_time (float): Time elapsed since the last step.

        Returns:
            mx.array: Updated state tensor.
        """
        v_pre = state

        # Precompute sensory neuron effects
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation *= self._params["sensory_sparsity_mask"]

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        w_numerator_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = mx.sum(sensory_w_activation, axis=1)

        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)

        # ODE unfolds
        for _ in range(self._ode_unfolds):
            def f(_, v_pre):
                w_activation = self._params["w"] * self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
                w_activation *= self._params["sparsity_mask"]

                rev_activation = w_activation * self._params["erev"]
                w_numerator = mx.sum(rev_activation, axis=1) + w_numerator_sensory
                w_denominator = mx.sum(w_activation, axis=1) + w_denominator_sensory

                numerator = cm_t * v_pre + self._params["gleak"] * self._params["vleak"] + w_numerator
                denominator = cm_t + self._params["gleak"] + w_denominator

                return numerator / (denominator + self._epsilon)

            v_pre = self.ode_solver(f, v_pre, 0, 1.0 / self._ode_unfolds)

        return v_pre

    def call(self, inputs, states):
        if self.solver == "rk4":
            return self._solve_rk4(inputs, states)
        elif self.solver == "euler":
            return self._solve_euler(inputs, states)
        else:
            return super().call(inputs, states)

    def __call__(self, inputs, states):
        """Make the cell callable."""
        return self.call(inputs, states)

    def get_config(self):
        """
        Returns a serialized configuration of the EnhancedLTCCell.
        """
        config = super().get_config()
        config.update({
            "solver": self.solver,
            "sparsity": self.sparsity,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstructs an EnhancedLTCCell from its serialized configuration.
        """
        base_config = LTCCell.from_config(config)
        return cls(
            **base_config,
            solver=config["solver"],
            sparsity=config["sparsity"],
            activation=config["activation"],
        )