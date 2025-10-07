"""Enhanced Liquid Time-Constant Cell implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from enum import Enum
from typing import Optional, Dict, Any, Callable, Union

from .ltc_cell import LTCCell
from .ode_solvers import rk4_solve, euler_solve, semi_implicit_solve


class ODESolver(Enum):
    """ODE solver types for continuous-time neural networks.
    
    Available solvers:
    - SEMI_IMPLICIT: Semi-implicit Euler method, good balance of stability and speed
    - EXPLICIT: Explicit Euler method, fastest but less stable
    - RUNGE_KUTTA: 4th order Runge-Kutta method, most accurate but computationally intensive
    """
    SEMI_IMPLICIT = "semi_implicit"
    EXPLICIT = "explicit"
    RUNGE_KUTTA = "rk4"


class ELTCCell(LTCCell):
    """Enhanced Liquid Time-Constant Cell (ELTC) for MLX.

    This class extends the LTCCell implementation by adding:
    - Configurable ODE solvers (Semi-implicit, Explicit, Runge-Kutta)
    - Sparsity constraints for adjacency matrices
    - Flexible activation functions
    - Dense layer transformations for input and recurrent connections

    The cell implements the following ODE:
        dy/dt = σ(Wx + Uh + b) - y
    where:
        - y is the cell state
        - x is the input
        - W is the input weight matrix
        - U is the recurrent weight matrix
        - b is the bias vector
        - σ is the activation function

    The ODE is solved using one of three methods:
    1. Semi-implicit Euler: Provides good stability and accuracy balance
    2. Explicit Euler: Fastest but may be unstable for stiff equations
    3. Runge-Kutta (RK4): Most accurate but computationally expensive
    """

    def __init__(
        self,
        wiring,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        solver: Union[str, ODESolver] = ODESolver.RUNGE_KUTTA,
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        initialization_ranges: Optional[Dict[str, Any]] = None,
        forget_gate_bias: float = 1.0,
        sparsity: float = 0.5,
        activation: Callable = mx.tanh,
        hidden_size: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the EnhancedLTCCell.

        Args:
            wiring: Neural wiring pattern
            input_mapping: Input mapping type ("affine" or "linear")
            output_mapping: Output mapping type ("affine" or "linear")
            solver: ODE solver type (ODESolver enum or string)
                - "semi_implicit": Semi-implicit Euler method
                - "explicit": Explicit Euler method
                - "rk4": 4th order Runge-Kutta method
            ode_unfolds: Number of ODE solver steps per time step
            epsilon: Small constant to avoid division by zero
            initialization_ranges: Ranges for parameter initialization
            forget_gate_bias: Bias for the forget gate
            sparsity: Sparsity level for adjacency matrices (0.0 to 1.0)
            activation: Activation function (default: tanh)
            hidden_size: Size of hidden state (optional, defaults to wiring units)
            **kwargs: Additional arguments passed to the base LTCCell
        """
        # Store ELTC-specific parameters before calling parent
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self._sparsity = sparsity
        self._eltc_activation = activation
        self._eltc_hidden_size = hidden_size
        
        # Filter kwargs to only pass what LTCCell accepts
        ltc_valid_kwargs = {'backbone_units', 'backbone_layers', 'backbone_dropout', 'initializer'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ltc_valid_kwargs}
        
        # Call parent - LTCCell only accepts: wiring, activation, backbone_units, backbone_layers, backbone_dropout, initializer
        super().__init__(
            wiring=wiring,
            activation=activation if isinstance(activation, str) else "tanh",
            **filtered_kwargs,
        )
        # Convert string solver type to enum if needed and store ELTC-specific params
        self.solver = solver if isinstance(solver, ODESolver) else ODESolver(solver)
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.sparsity = self._sparsity
        self.activation = self._eltc_activation
        self.hidden_size = self._eltc_hidden_size or self.units

        # Initialize dense layers based on mapping type
        if self.input_mapping == "affine":
            self.input_dense = nn.Linear(self.input_size, self.hidden_size)
        if self.output_mapping == "affine":
            self.recurrent_dense = nn.Linear(self.hidden_size, self.hidden_size)

    def build(self):
        """Initialize parameters and apply sparsity."""
        super().build()
        self._apply_sparsity()

    def _apply_sparsity(self):
        """Apply sparsity constraints to the adjacency matrix.
        
        Creates a binary mask using Bernoulli distribution where:
        - 1 indicates a connection is kept
        - 0 indicates a connection is dropped
        The sparsity parameter controls the probability of dropping connections.
        """
        if not hasattr(self, 'weight'):
            return  # Skip if params not yet initialized
            
        # Use bernoulli for sparsity mask
        mask = mx.random.bernoulli(1 - self.sparsity, self.weight.shape)
        self.weight = self.weight * mask

    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODEs with enhanced solvers.

        The ODE being solved is:
            dy/dt = σ(Wx + Uh + b) - y

        The solution process:
        1. Compute dense transformations for input and recurrent connections
        2. Define the ODE function f(t, y) = σ(net_input) - y
        3. Apply the selected solver with proper time stepping
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            state: Current state tensor of shape [batch_size, hidden_size]
            elapsed_time: Time elapsed since last update

        Returns:
            Updated state tensor of shape [batch_size, hidden_size]
        """
        # Apply dense transformations
        input_proj = self.input_dense(inputs)
        recurrent_proj = self.recurrent_dense(state)
        net_input = input_proj + recurrent_proj

        # Define ODE function
        def f(t, y):
            return self.activation(net_input) - y

        # Apply selected solver
        dt = elapsed_time / self.ode_unfolds
        v_pre = state

        for _ in range(self.ode_unfolds):
            if self.solver == ODESolver.SEMI_IMPLICIT:
                v_pre = semi_implicit_solve(f, v_pre, dt)
            elif self.solver == ODESolver.EXPLICIT:
                v_pre = euler_solve(f, v_pre, dt)
            elif self.solver == ODESolver.RUNGE_KUTTA:
                v_pre = rk4_solve(f, v_pre, 0, dt)
            else:
                raise ValueError(f"Unsupported solver type: {self.solver}")

        return v_pre

    def __call__(self, inputs, state=None, time=1.0):
        """Process one time step through the cell.

        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            state: Optional initial state of shape [batch_size, hidden_size]
            time: Time step size (default: 1.0)

        Returns:
            Tuple of:
            - output: Processed output of shape [batch_size, output_dim]
            - new_state: Updated state as list [state] of shape [batch_size, hidden_size]
        """
        batch_size = inputs.shape[0]
        if state is None:
            state = mx.zeros((batch_size, self.hidden_size))

        new_state = self._ode_solver(inputs, state, time)
        output = self.activation(new_state)

        if self.wiring.output_dim != self.hidden_size:
            output = output[:, :self.wiring.output_dim]

        return output, [new_state]

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "solver": self.solver.value,
            "sparsity": self.sparsity,
            "activation": self.activation,
            "hidden_size": self.hidden_size,
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
            hidden_size=config.get("hidden_size"),
        )
