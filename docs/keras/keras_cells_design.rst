Keras Cell Implementations Design
=================================

CfCCell Implementation
----------------------

Overview
~~~~~~~~

The CfCCell implements the Closed-form Continuous-time mechanics,
supporting multiple operation modes and efficient backbone integration.

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

class CfCCell(LiquidCell):
    def __init__(
        self,
        wiring,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
    ):
        """Initialize CfC cell.

        Args:
            wiring: Neural circuit wiring
            mode: Operation mode ('default', 'pure', 'no_gate')
            activation: Activation function
            backbone_units: Backbone layer sizes
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
        """
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        self.mode = mode

Key Components
~~~~~~~~~~~~~~

1. Build Method

.. code:: python

def build(self, input_shape):
    """Build cell parameters."""
    super().build(input_shape)

    # Get effective input dimension
    if self.backbone is not None:
        input_dim = self.backbone_output_dim
    else:
        input_dim = self.input_size + self.hidden_size

    # Initialize transformation weights
    self.ff1_kernel = self.add_weight(
        shape=(input_dim, self.hidden_size),
        initializer="glorot_uniform",
        name="ff1_weight"
    )
    self.ff1_bias = self.add_weight(
        shape=(self.hidden_size,),
        initializer="zeros",
        name="ff1_bias"
    )

    if self.mode == "pure":
        self._build_pure_mode()
    else:
        self._build_gated_mode(input_dim)

2. Call Method

.. code:: python

def call(self, inputs, states, training=None):
    """Process one step."""
    # Get time input
    if isinstance(inputs, (list, tuple)):
        inputs, time = inputs
    else:
        time = 1.0

    # Process input
    x = keras.layers.concatenate([inputs, states[0]])
    if self.backbone is not None:
        x = self.backbone(x, training=training)

    # Apply transformations
    if self.mode == "pure":
        return self._pure_step(x, time)
    else:
        return self._gated_step(x, time)

LTCCell Implementation
----------------------

.. _overview-1:

Overview
~~~~~~~~

The LTCCell implements the Liquid Time-Constant mechanics, providing
biologically-inspired dynamics with time-dependent processing.

.. _class-structure-1:

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

class LTCCell(LiquidCell):
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
    ):
        """Initialize LTC cell.

        Args:
            wiring: Neural circuit wiring
            activation: Activation function
            backbone_units: Backbone layer sizes
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
        """
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )

.. _key-components-1:

Key Components
~~~~~~~~~~~~~~

1. Build Method

.. code:: python

def build(self, input_shape):
    """Build cell parameters."""
    super().build(input_shape)

    # Get effective input dimension
    if self.backbone is not None:
        input_dim = self.backbone_output_dim
    else:
        input_dim = self.input_size + self.hidden_size

    # Initialize transformation weights
    self.kernel = self.add_weight(
        shape=(input_dim, self.hidden_size),
        initializer="glorot_uniform",
        name="kernel"
    )
    self.bias = self.add_weight(
        shape=(self.hidden_size,),
        initializer="zeros",
        name="bias"
    )

    # Initialize time constant network
    self.tau_kernel = keras.layers.Dense(
        self.hidden_size,
        name="tau_kernel"
    )

2. Call Method

.. code:: python

def call(self, inputs, states, training=None):
    """Process one step."""
    # Get time input
    if isinstance(inputs, (list, tuple)):
        inputs, time = inputs
    else:
        time = 1.0

    # Process input
    x = keras.layers.concatenate([inputs, states[0]])
    if self.backbone is not None:
        x = self.backbone(x, training=training)

    # Compute delta term
    d = keras.backend.dot(x, self.kernel) + self.bias

    # Compute time constants
    tau = keras.backend.exp(self.tau_kernel(x))

    # Update state
    new_state = states[0] + time * (-states[0] + d) / tau

    # Apply activation
    output = self.activation(new_state)

    return output, [new_state]

Implementation Notes
--------------------

1. Mode Handling (CfC)
~~~~~~~~~~~~~~~~~~~~~~

1. Pure Mode

.. code:: python

def _build_pure_mode(self):
    """Build pure mode parameters."""
    self.w_tau = self.add_weight(
        shape=(1, self.hidden_size),
        initializer="zeros",
        name="w_tau"
    )
    self.A = self.add_weight(
        shape=(1, self.hidden_size),
        initializer="ones",
        name="A"
    )

def _pure_step(self, x, time):
    """Execute pure mode step."""
    ff1 = keras.backend.dot(x, self.ff1_kernel) + self.ff1_bias
    new_state = (
        -self.A

        * keras.backend.exp(-time * (keras.backend.abs(self.w_tau) + keras.backend.abs(ff1)))
        * ff1

        + self.A
    )
    return new_state, [new_state]

2. Gated Mode

.. code:: python

def _build_gated_mode(self, input_dim):
    """Build gated mode parameters."""
    self.ff2_kernel = self.add_weight(
        shape=(input_dim, self.hidden_size),
        initializer="glorot_uniform",
        name="ff2_weight"
    )
    self.ff2_bias = self.add_weight(
        shape=(self.hidden_size,),
        initializer="zeros",
        name="ff2_bias"
    )
    self.time_a = keras.layers.Dense(self.hidden_size, name="time_a")
    self.time_b = keras.layers.Dense(self.hidden_size, name="time_b")

def _gated_step(self, x, time):
    """Execute gated mode step."""
    ff1 = keras.backend.dot(x, self.ff1_kernel) + self.ff1_bias
    ff2 = keras.backend.dot(x, self.ff2_kernel) + self.ff2_bias

    t_a = self.time_a(x)
    t_b = self.time_b(x)
    t_interp = keras.backend.sigmoid(-t_a * time + t_b)

    if self.mode == "no_gate":
        new_state = ff1 + t_interp * ff2
    else:
        new_state = ff1 * (1.0 - t_interp) + t_interp * ff2

    return new_state, [new_state]

2. Time Processing
~~~~~~~~~~~~~~~~~~

1. Time Input Handling

.. code:: python

def _process_time(self, time):
    """Process time input."""
    if isinstance(time, (int, float)):
        return keras.backend.constant(time)
    return keras.backend.cast(time, dtype=self.dtype)

2. Time Broadcasting

.. code:: python

def _broadcast_time(self, time, batch_size):
    """Broadcast time for batch processing."""
    if keras.backend.ndim(time) == 0:
        return keras.backend.reshape(time, (1, 1))
    elif keras.backend.ndim(time) == 1:
        return keras.backend.reshape(time, (-1, 1))
    return time

3. Training Support
~~~~~~~~~~~~~~~~~~~

1. Dropout Handling

.. code:: python

def _apply_dropout(self, x, training):
    """Apply dropout during training."""
    if training and self.backbone_dropout > 0:
        return keras.layers.Dropout(self.backbone_dropout)(x)
    return x

2. State Updates

.. code:: python

def _update_state(self, state, new_state, training):
    """Update state with optional noise during training."""
    if training and hasattr(self, 'state_noise'):
        new_state += keras.backend.random_normal(
            keras.backend.shape(new_state),
            mean=0.0,
            stddev=self.state_noise
        )
    return new_state

This design provides detailed implementations for both CfC and LTC cells
in Keras, maintaining compatibility with the framework while adding our
improved functionality.
