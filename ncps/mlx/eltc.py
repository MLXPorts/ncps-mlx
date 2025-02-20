"""Enhanced Liquid Time-Constant (ELTC) RNN implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any

from .base import LiquidRNN
from .liquid_utils import get_activation
from .eltc_cell import ELTCCell, ODESolver


class ELTC(LiquidRNN):
    """An Enhanced Liquid Time-Constant (ELTC) RNN layer.
    
    The input is a sequence of shape NLD or LD where:
    - N is the optional batch dimension
    - L is the sequence length
    - D is the input's feature dimension

    The network processes sequences using an enhanced version of the Liquid 
    Time-Constant cell that includes:
    - Configurable ODE solvers (Semi-implicit, Explicit, Runge-Kutta)
    - Dense transformations for input and recurrent connections
    - Sparsity control for network connectivity
    - Flexible activation functions
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        solver: Union[str, ODESolver] = ODESolver.RUNGE_KUTTA,
        ode_unfolds: int = 6,
        activation: str = "tanh",
        bias: bool = True,
        cell_clip: Optional[float] = None,
        epsilon: float = 1e-8,
        sparsity: float = 0.5,
        return_sequences: bool = True,
        return_state: bool = False,
        bidirectional: bool = False,
        merge_mode: Optional[str] = None,
    ):
        """Initialize the ELTC layer.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            solver: ODE solver type (ODESolver enum or string)
                - "semi_implicit": Semi-implicit Euler method
                - "explicit": Explicit Euler method
                - "rk4": 4th order Runge-Kutta method
            ode_unfolds: Number of ODE solver steps per time step
            activation: Activation function name
            bias: Whether to use bias in dense layers
            cell_clip: Optional value for gradient clipping
            epsilon: Small constant to avoid division by zero
            sparsity: Sparsity level for adjacency matrices (0.0 to 1.0)
            return_sequences: Whether to return the full sequence
            return_state: Whether to return the final state
            bidirectional: Whether to process sequences bidirectionally
            merge_mode: How to merge bidirectional outputs
                - "concat": Concatenate forward and backward outputs
                - "sum": Add forward and backward outputs
                - "mul": Multiply forward and backward outputs
                - "ave": Average forward and backward outputs
        """
        # Create wiring for the cell
        from .wirings import FullyConnected
        wiring = FullyConnected(hidden_size, hidden_size)
        wiring.build(input_size)
        
        # Create the ELTC cell
        cell = ELTCCell(
            wiring=wiring,
            input_mapping="affine",
            output_mapping="affine",
            solver=solver,
            ode_unfolds=ode_unfolds,
            activation=get_activation(activation),
            bias=bias,
            cell_clip=cell_clip,
            epsilon=epsilon,
            sparsity=sparsity,
            hidden_size=hidden_size,
        )
        
        # Initialize the RNN with the cell
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
        )
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver = solver if isinstance(solver, ODESolver) else ODESolver(solver)
        self.ode_unfolds = ode_unfolds
        self.activation = activation
        self.bias = bias
        self.cell_clip = cell_clip
        self.epsilon = epsilon
        self.sparsity = sparsity

    def _extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"solver={self.solver.value}, "
            f"ode_unfolds={self.ode_unfolds}, "
            f"sparsity={self.sparsity}, "
            f"bias={self.bias}"
        )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "solver": self.solver.value,
            "ode_unfolds": self.ode_unfolds,
            "activation": self.activation,
            "bias": self.bias,
            "cell_clip": self.cell_clip,
            "epsilon": self.epsilon,
            "sparsity": self.sparsity,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ELTC":
        """Create instance from configuration."""
        return cls(**config)