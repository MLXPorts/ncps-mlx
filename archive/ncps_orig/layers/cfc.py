"""Closed-form Continuous-time (CfC) cell implementation."""

from keras import ops, layers, activations, initializers, regularizers, constraints, random
from typing import Optional, List, Any, Union, Tuple

from .base import BackboneLayerCell


class CfCCell(BackboneLayerCell):
    """A Closed-form Continuous-time cell.
    
    Args:
        units: Positive integer, dimensionality of the output space.
        mode: Either "default", "pure" (direct solution approximation),
            or "no_gate" (without second gate).
        activation: Activation function to use.
        use_bias: Boolean, whether to use bias vectors.
        kernel_initializer: Initializer for input kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for input kernels.
        bias_regularizer: Regularizer for bias vectors.
        kernel_constraint: Constraint for input kernels.
        bias_constraint: Constraint for bias vectors.
        dropout: Float between 0 and 1. Dropout rate.
        recurrent_dropout: Float between 0 and 1. Recurrent dropout rate.
        seed: Random seed for dropout.
        **kwargs: Additional keyword arguments for the base layer.
    """
    
    def __init__(
        self,
        units: int,
        mode: str = "default",
        activation: Union[str, layers.Layer] = "tanh",
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(units, **kwargs)
        
        # Validate mode
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        
        self.mode = mode
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        
        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        # Dropout
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed=seed)
        
        # Reset dropout masks
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        # State size is set in parent
        self.state_size = self.units
        self.output_size = self.units
    
    def build(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> None:
        """Build the cell weights.
        
        Args:
            input_shape: Tuple of integers, the input shape.
        """
        def _get_input_dim(shape):
            """Extract input dimension from shape."""
            if shape is None:
                raise ValueError("Input shape cannot be None")
            if isinstance(shape, int):
                return shape
            if not isinstance(shape, (list, tuple)):
                raise ValueError(f"Invalid input shape: {shape}")
            if len(shape) < 2:
                raise ValueError(f"Input shape must have at least 2 dimensions: {shape}")
            return shape[-1]
        
        # Handle input shape
        if isinstance(input_shape, list):
            # [inputs, time] format
            input_shape = input_shape[0]
        
        # Get input dimension
        input_dim = _get_input_dim(input_shape)
        
        # Calculate concatenated input size
        concat_dim = input_dim + self.units
        
        super().build(input_shape)
        
        # Main transformation weights for concatenated input
        self.kernel = self.add_weight(
            shape=(concat_dim, self.units),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        
        # Mode-specific weights
        if self.mode == "pure":
            self._build_pure_mode()
        else:
            self._build_gated_mode()
        
        self.built = True
    
    def _build_pure_mode(self) -> None:
        """Initialize pure mode weights."""
        self.w_tau = self.add_weight(
            shape=(1, self.units),
            name="w_tau",
            initializer="zeros"
        )
        self.A = self.add_weight(
            shape=(1, self.units),
            name="A",
            initializer="ones"
        )
    
    def _build_gated_mode(self) -> None:
        """Initialize gated mode weights."""
        self.gate_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="gate_kernel",
            initializer=self.kernel_initializer,
        )
        
        if self.use_bias:
            self.gate_bias = self.add_weight(
                shape=(self.units,),
                name="gate_bias",
                initializer=self.bias_initializer,
            )
    
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
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError("Expected [inputs, time] when passing time")
            x = inputs[0]  # Input tensor
            t = inputs[1]  # Time tensor
            t = ops.reshape(t, [-1, 1])
        else:
            x = inputs
            t = 1.0
        
        # Get current state
        h_prev = states[0]
        
        # Create and apply dropout masks
        if training:
            if self.dropout > 0:
                dp_mask = self._create_dropout_mask(x, self.dropout)
                x = x * dp_mask
            if self.recurrent_dropout > 0:
                rec_dp_mask = self._create_dropout_mask(h_prev, self.recurrent_dropout)
                h_prev = h_prev * rec_dp_mask
        
        # Combine inputs and state
        concat = ops.concatenate([x, h_prev], axis=-1)
        
        # Apply main transformation
        h = ops.matmul(concat, self.kernel)
        if self.use_bias:
            h = h + self.bias
        
        # Mode-specific processing
        if self.mode == "pure":
            new_state = self._pure_step(h, t)
        else:
            new_state = self._gated_step(h, h_prev, t)
        
        # Apply activation
        output = self.activation(new_state)
        
        # Reset dropout masks
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        return output, [new_state]
    
    def _create_dropout_mask(self, inputs, rate):
        """Create dropout mask."""
        ones = ops.ones_like(inputs)
        return random.dropout(
            ones, rate=rate, seed=self.seed_generator
        )
    
    def _pure_step(
        self,
        h: keras.KerasTensor,
        t: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Execute pure mode step."""
        return (
            -self.A 
            * ops.exp(-t * (ops.abs(self.w_tau) + ops.abs(h))) 
            * h 
            + self.A
        )
    
    def _gated_step(
        self,
        h: keras.KerasTensor,
        h_prev: keras.KerasTensor,
        t: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Execute gated mode step."""
        # Compute gate with direct time scaling
        gate = ops.matmul(h_prev, self.gate_kernel)
        if self.use_bias:
            gate = gate + self.gate_bias
        gate = keras.activations.sigmoid(-t * gate)
        
        # Scale state update by time
        if self.mode == "no_gate":
            return h + t * gate * h_prev
        else:
            return h * (1.0 - t * gate) + t * gate * h_prev
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "mode": self.mode,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        })
        return config