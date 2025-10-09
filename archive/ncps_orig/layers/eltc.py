"""Enhanced Linear Time-Constant (ELTC) cell implementation."""

import enum
import keras
from keras import ops, layers
from typing import Optional, List, Any, Union, Tuple, Dict, Callable

from .base import BackboneLayerCell


class ODESolver(str, enum.Enum):
    """ODE solver types for continuous-time neural networks."""
    SEMI_IMPLICIT = "semi_implicit"
    EXPLICIT = "explicit"
    RUNGE_KUTTA = "rk4"


@keras.saving.register_keras_serializable(package="ncps")
class ELTCCell(BackboneLayerCell):
    """An Enhanced Linear Time-Constant cell.
    
    This extends the basic LTC cell with:
    - Configurable ODE solvers
    - Sparsity constraints
    - Flexible activation functions
    - Dense layer transformations
    
    Args:
        units: Positive integer, dimensionality of the output space.
        solver: ODE solver type ("semi_implicit", "explicit", "rk4").
        ode_unfolds: Number of ODE solver steps per time step.
        sparsity: Sparsity level for weight matrices (0.0 to 1.0).
        activation: Activation function to use.
        hidden_size: Optional size for hidden state.
        backbone_units: Number of units in backbone layers.
        backbone_layers: Number of backbone layers.
        backbone_dropout: Dropout rate in backbone layers.
        **kwargs: Additional keyword arguments for the base layer.
    """
    
    def __init__(
        self,
        units: int,
        solver: Union[str, ODESolver] = ODESolver.RUNGE_KUTTA,
        ode_unfolds: int = 6,
        sparsity: float = 0.5,
        activation: Union[str, layers.Layer] = "tanh",
        hidden_size: Optional[int] = None,
        backbone_units: Optional[int] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(units, **kwargs)
        
        # Convert string solver type to enum if needed
        self.solver = solver if isinstance(solver, ODESolver) else ODESolver(solver)
        self.ode_unfolds = ode_unfolds
        self.sparsity = sparsity
        self.activation = keras.activations.get(activation)
        self.hidden_size = hidden_size or units
        
        # Backbone configuration
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone_fn = None
        
        # Initialize dense transformations
        self.input_dense = None
        self.recurrent_dense = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the cell weights.
        
        Args:
            input_shape: Tuple of integers, the input shape.
        """
        super().build(input_shape)
        
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build backbone if needed
        if self.backbone_layers > 0 and self.backbone_units:
            backbone_layers = []
            for i in range(self.backbone_layers):
                backbone_layers.append(
                    layers.Dense(
                        self.backbone_units,
                        self.activation,
                        name=f"backbone{i}"
                    )
                )
                if self.backbone_dropout > 0:
                    backbone_layers.append(
                        layers.Dropout(self.backbone_dropout)
                    )
            
            self.backbone_fn = keras.Sequential(backbone_layers)
            self.backbone_fn.build((None, self.hidden_size + input_dim))
            cat_shape = self.backbone_units
        else:
            cat_shape = self.hidden_size + input_dim
        
        # Initialize dense transformations
        self.input_dense = layers.Dense(self.hidden_size)
        self.recurrent_dense = layers.Dense(self.hidden_size)
        
        self.input_dense.build((None, input_dim))
        self.recurrent_dense.build((None, self.hidden_size))
        
        # Initialize sparsity mask
        self.sparsity_mask = keras.random.bernoulli(
            shape=(cat_shape, self.hidden_size),
            p=1.0 - self.sparsity,
            dtype=self.compute_dtype
        )
    
    def _solve_ode(
        self,
        f: Callable,
        y: keras.KerasTensor,
        dt: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Solve ODE using selected solver.
        
        Args:
            f: ODE function dy/dt = f(t, y)
            y: Current state
            dt: Time step size
            
        Returns:
            Updated state
        """
        if self.solver == ODESolver.SEMI_IMPLICIT:
            # Semi-implicit Euler method
            f_eval = f(0, y)
            return y + dt * f_eval
        
        elif self.solver == ODESolver.EXPLICIT:
            # Explicit Euler method
            f_eval = f(0, y)
            return y + dt * f_eval
        
        else:  # ODESolver.RUNGE_KUTTA
            # 4th order Runge-Kutta method
            k1 = f(0, y)
            k2 = f(dt/2, y + dt*k1/2)
            k3 = f(dt/2, y + dt*k2/2)
            k4 = f(dt, y + dt*k3)
            return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        states: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """Process one timestep.
        
        Args:
            inputs: Input tensor or list of [input, time] tensors.
            states: List of state tensors.
            training: Whether in training mode.
            
        Returns:
            Tuple of (output tensor, list of new state tensors).
        """
        # Handle time input
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = ops.reshape(t, [-1, 1])
        else:
            t = 1.0
        
        # Get current state
        h_prev = states[0]
        
        # Apply dense transformations
        x_proj = self.input_dense(inputs)
        h_proj = self.recurrent_dense(h_prev)
        
        # Combine and apply backbone if present
        if self.backbone_fn is not None:
            combined = layers.Concatenate()([x_proj, h_proj])
            net_input = self.backbone_fn(combined, training=training)
        else:
            net_input = x_proj + h_proj
        
        # Apply sparsity
        if hasattr(self, 'sparsity_mask'):
            net_input = net_input * self.sparsity_mask
        
        # Define ODE function
        def f(_, y):
            return self.activation(net_input) - y
        
        # Solve ODE
        dt = t / self.ode_unfolds
        h = h_prev
        
        for _ in range(self.ode_unfolds):
            h = self._solve_ode(f, h, dt)
        
        return h, [h]
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "solver": self.solver.value,
            "ode_unfolds": self.ode_unfolds,
            "sparsity": self.sparsity,
            "activation": keras.activations.serialize(self.activation),
            "hidden_size": self.hidden_size,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout
        })
        return config