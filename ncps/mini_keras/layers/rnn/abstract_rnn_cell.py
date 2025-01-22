from ncps.mini_keras import backend
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.layer import Layer
from abc import ABC, abstractmethod

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = backend.shape(inputs)[0]  # Changed from tf.shape
        dtype = inputs.dtype
    return backend.zeros([batch_size, cell.state_size], dtype=dtype)  # Changed from tf.zeros

@keras_mini_export("ncps.mini_keras.layers.AbstractRNNCell")
class AbstractRNNCell(Layer, ABC):
    """Abstract base class for RNN cells.

    This class defines the interface for custom RNN cells. Subclasses must implement
    the `state_size`, `output_size`, and `call` methods.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        **kwargs: Additional keyword arguments.

    Subclasses must implement the following methods:
        - `state_size`: Returns the size of the state(s) used by the cell.
        - `output_size`: Returns the size of the output produced by the cell.
        - `call(inputs, states)`: Defines the computation from inputs and states to outputs and new states.
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    @property
    @abstractmethod
    def state_size(self):
        """Size of state(s) used by this cell."""
        pass

    @property
    @abstractmethod
    def output_size(self):
        """Size of outputs produced by this cell."""
        pass

    @abstractmethod
    def call(self, inputs, states):
        """Defines the computation from inputs and states to outputs and new states.

        The function that contains the logic for one RNN step calculation.

        Args:
            inputs: The input tensor, which is a slice from the overall RNN input by
                the time dimension (usually the second dimension).
            states: The state tensor from previous step, which has the same shape
                as `(batch, state_size)`. In the case of timestep 0, it will be the
                initial state user specified, or zero filled tensor otherwise.

        Returns:
            A tuple of two tensors:
                1. Output tensor for the current timestep, with size `output_size`.
                2. State tensor for next step, which has the shape of `state_size`.
        """
        pass

    def build(self, input_shape):
        # Define weights and biases here
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "recurrent_constraint": self.recurrent_constraint,
            "bias_constraint": self.bias_constraint,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def reset_recurrent_dropout_mask(self):
        """Reset the cached recurrent dropout masks if any."""
        pass  # Implement if necessary
