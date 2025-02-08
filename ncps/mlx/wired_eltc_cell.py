from .eltc_cell import EnhancedLTCCell
from ncps.wirings import Wiring
from ncps.mini_keras import layers
from ncps.mini_keras.ops.ode import SOLVERS
import ncps

@ncps.mini_keras.utils.register_keras_serializable(package="ncps", name="WiredEnhancedLTCCell")
class WiredEnhancedLTCCell(layers.RNNCell):
    """Enhanced Liquid Time-Constant Cell with wiring support."""

    def __init__(
        self,
        wiring: Wiring,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        solver: str = "rk4",
        ode_unfolds: int = 6,
        mixed_memory: bool = False,
        backbone_units: int = 0,
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        if backbone_units > 0 and wiring.is_sparse():
            raise ValueError("If sparsity is set, then no backbone is allowed")
            
        self.wiring = wiring
        self.solver = solver
        self.ode_unfolds = ode_unfolds
        self.mixed_memory = mixed_memory
        self.backbone_units = backbone_units
        self.epsilon = epsilon
        
        if solver not in SOLVERS:
            raise ValueError(f"Unknown solver {solver}. Available solvers: {list(SOLVERS.keys())}")
        
        self.solve = SOLVERS[solver]

    @property
    def state_size(self):
        return (self.wiring.units, self.wiring.units) if self.mixed_memory else self.wiring.units

    def build(self, input_shape):
        # ...existing build code using mini_keras layers...
        pass

    def call(self, inputs, states):
        """Forward pass using mini_keras infrastructure."""
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

    def _ode_step(self, inputs, state):
        """Single ODE step using configured solver."""
        def ode_fn(t, y):
            return -y + inputs
            
        v_pre = state
        dt = 1.0 / self.ode_unfolds
        
        for _ in range(self.ode_unfolds):
            v_pre = self.solve(ode_fn, v_pre, 0, dt)
            
        return v_pre

    def get_config(self):
        config = super().get_config()
        config.update({
            "wiring": self.wiring.get_config(),
            "solver": self.solver,
            "ode_unfolds": self.ode_unfolds,
            "mixed_memory": self.mixed_memory,
            "backbone_units": self.backbone_units,
            "epsilon": self.epsilon,
        })
        return config
