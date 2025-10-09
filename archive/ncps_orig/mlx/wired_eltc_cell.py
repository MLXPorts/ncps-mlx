"""Wired Enhanced LTC cell implementation in MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any, Union, Tuple

from .eltc_cell import ELTCCell
from .ode_solvers import rk4_solve, euler_solve, semi_implicit_solve


class WiredELTCCell(nn.Module):
    """Enhanced Liquid Time-Constant Cell with wiring support."""

    def __init__(
        self,
        wiring,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        solver: str = "rk4",
        ode_unfolds: int = 6,
        mixed_memory: bool = False,
        backbone_units: int = 0,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """
        Initialize WiredELTCCell.
        
        Args:
            wiring: Neural wiring pattern
            input_mapping: Input mapping type ("affine" or "linear")
            output_mapping: Output mapping type ("affine" or "linear")
            solver: ODE solver type ("rk4", "euler", "semi_implicit")
            ode_unfolds: Number of ODE unfolds per step
            mixed_memory: Whether to use mixed memory mode
            backbone_units: Number of backbone layer units
            epsilon: Small constant for numerical stability
            **kwargs: Additional arguments
        """
        super().__init__()
        if backbone_units > 0 and wiring.is_sparse():
            raise ValueError("If sparsity is set, then no backbone is allowed")
            
        self.wiring = wiring
        self.solver = solver
        self.ode_unfolds = ode_unfolds
        self.mixed_memory = mixed_memory
        self.backbone_units = backbone_units
        self.epsilon = epsilon
        
        allowed_solvers = ["rk4", "euler", "semi_implicit"]
        if solver not in allowed_solvers:
            raise ValueError(f"Unknown solver {solver}. Available solvers: {allowed_solvers}")
        
        # Create base cell
        self.cell = ELTCCell(
            wiring=wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            solver=solver,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            backbone_units=backbone_units,
        )

    @property
    def state_size(self) -> Union[int, Tuple[int, int]]:
        """Get state size."""
        return (self.wiring.units, self.wiring.units) if self.mixed_memory else self.wiring.units

    def ode_solver(self, f, y0, t0, dt):
        """
        Solve ODEs using the configured solver.
        
        Args:
            f: ODE function dy/dt = f(t, y)
            y0: Initial state
            t0: Initial time
            dt: Time step size
            
        Returns:
            Updated state
        """
        if self.solver == "rk4":
            return rk4_solve(f, y0, t0, dt)
        elif self.solver == "euler":
            return euler_solve(f, y0, dt)
        elif self.solver == "semi_implicit":
            return semi_implicit_solve(f, y0, dt)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver}")

    def _ode_step(self, inputs: mx.array, state: mx.array) -> mx.array:
        """
        Single ODE step using configured solver.
        
        Args:
            inputs: Input tensor
            state: Current state
            
        Returns:
            Updated state
        """
        def ode_fn(t, y):
            return -y + inputs
            
        v_pre = state
        dt = 1.0 / self.ode_unfolds
        
        for _ in range(self.ode_unfolds):
            v_pre = self.ode_solver(ode_fn, v_pre, 0, dt)
            
        return v_pre

    def __call__(
        self,
        inputs: mx.array,
        states: Union[mx.array, Tuple[mx.array, mx.array]],
        time_delta: Optional[Union[float, mx.array]] = None,
    ) -> Tuple[mx.array, Union[mx.array, Tuple[mx.array, mx.array]]]:
        """
        Process one time step.
        
        Args:
            inputs: Input tensor
            states: Current state(s)
            time_delta: Optional time step size
            
        Returns:
            Tuple of (output, new_state(s))
        """
        if self.mixed_memory:
            h, c = states
            new_h = self._ode_step(inputs, h)
            new_c = h  # Update cell state
            output = self.wiring.get_outputs(new_h)
            return output, (new_h, new_c)
        else:
            new_h = self._ode_step(inputs, states)
            output = self.wiring.get_outputs(new_h)
            return output, new_h

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        return {
            "wiring": self.wiring,
            "solver": self.solver,
            "ode_unfolds": self.ode_unfolds,
            "mixed_memory": self.mixed_memory,
            "backbone_units": self.backbone_units,
            "epsilon": self.epsilon,
            "cell": self.cell.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load from state dictionary."""
        self.wiring = state_dict["wiring"]
        self.solver = state_dict["solver"]
        self.ode_unfolds = state_dict["ode_unfolds"]
        self.mixed_memory = state_dict["mixed_memory"]
        self.backbone_units = state_dict["backbone_units"]
        self.epsilon = state_dict["epsilon"]
        self.cell.load_state_dict(state_dict["cell"])
