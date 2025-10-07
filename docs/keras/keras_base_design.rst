Keras Base Classes Design
=========================

LiquidCell Base Class
---------------------

Overview
~~~~~~~~

The LiquidCell base class will serve as the foundation for all liquid
neural network cells in the Keras implementation. It will provide common
functionality for backbone networks, dimension tracking, and state
management.

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

class LiquidCell(keras.layers.Layer):
    def __init__(
        self,
        wiring,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
    ):
        pass

Key Components
~~~~~~~~~~~~~~

1. Initialization

- Wiring configuration
- Backbone network setup
- Dimension tracking
- State initialization

2. Backbone Network

.. code:: python

def build_backbone(self):
    """Build backbone network layers."""
    layers = []
    current_dim = self.input_size + self.hidden_size

    for i, units in enumerate(self.backbone_units):
        layers.append(keras.layers.Dense(units))
        layers.append(self.activation)
        if self.backbone_dropout > 0:
            layers.append(keras.layers.Dropout(self.backbone_dropout))

    return keras.Sequential(layers) if layers else None

3. Dimension Handling

.. code:: python

def build(self, input_shape):
    """Build cell parameters."""
    # Calculate dimensions
    self.input_size = input_shape[-1]
    self.hidden_size = self.wiring.units

    # Build backbone if specified
    if self.backbone_layers > 0:
        self.backbone = self.build_backbone()
        self.backbone_output_dim = self.backbone_units[-1]
    else:
        self.backbone_output_dim = self.input_size + self.hidden_size

4. State Management

.. code:: python

@property
def state_size(self):
    """Return state size for RNN."""
    return self.hidden_size

def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Create initial state."""
    return [
        keras.backend.zeros((batch_size, self.hidden_size))
    ]

Interface Methods
~~~~~~~~~~~~~~~~~

1. Call Method

.. code:: python

def call(self, inputs, states, training=None):
    """Process one step with the cell."""
    raise NotImplementedError()

2. Configuration

.. code:: python

def get_config(self):
    """Get configuration for serialization."""
    config = super().get_config()
    config.update({
        'wiring': self.wiring.get_config(),
        'activation': self.activation_name,
        'backbone_units': self.backbone_units,
        'backbone_layers': self.backbone_layers,
        'backbone_dropout': self.backbone_dropout,
    })
    return config

LiquidRNN Base Class
--------------------

.. _overview-1:

Overview
~~~~~~~~

The LiquidRNN base class will provide high-level RNN functionality,
wrapping LiquidCell instances and handling sequence processing.

.. _class-structure-1:

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

class LiquidRNN(keras.layers.RNN):
    def __init__(
        self,
        cell,
        return_sequences=True,
        return_state=False,
        bidirectional=False,
        merge_mode="concat",
        **kwargs
    ):
        pass

.. _key-components-1:

Key Components
~~~~~~~~~~~~~~

1. Initialization

.. code:: python

def __init__(self, cell, **kwargs):
    """Initialize RNN with liquid cell."""
    super().__init__(
        cell,
        return_sequences=kwargs.pop('return_sequences', True),
        return_state=kwargs.pop('return_state', False),
        **kwargs
    )
    self.supports_masking = True

2. Bidirectional Support

.. code:: python

def _make_bidirectional(self):
    """Create bidirectional wrapper."""
    return keras.layers.Bidirectional(
        self,
        merge_mode=self.merge_mode,
        backward_layer=type(self)(
            type(self.cell).from_config(self.cell.get_config()),
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            go_backwards=True,
        )
    )

3. Time Processing

.. code:: python

def _process_time(self, inputs, time_steps=None):
    """Process time inputs."""
    if time_steps is None:
        return 1.0
    return keras.backend.cast(time_steps, dtype=self.dtype)

.. _interface-methods-1:

Interface Methods
~~~~~~~~~~~~~~~~~

1. Call Method

.. code:: python

def call(self, inputs, mask=None, training=None, initial_state=None):
    """Process input sequence."""
    # Handle inputs
    if isinstance(inputs, (list, tuple)):
        inputs, *rest = inputs
        time_steps = rest[0] if rest else None
    else:
        time_steps = None

    # Process time
    time = self._process_time(inputs, time_steps)

    # Call parent with processed inputs
    return super().call(
        inputs,
        mask=mask,
        training=training,
        initial_state=initial_state,
        constants=[time] if time_steps is not None else None
    )

2. Configuration

.. code:: python

def get_config(self):
    """Get configuration for serialization."""
    config = super().get_config()
    config.update({
        'cell': {
            'class_name': self.cell.__class__.__name__,
            'config': self.cell.get_config()
        }
    })
    return config

Implementation Notes
--------------------

1. Keras Specifics
~~~~~~~~~~~~~~~~~~

- Use Keras backend operations
- Follow Keras layer conventions
- Support Keras masking
- Handle Keras training modes

2. Dimension Handling
~~~~~~~~~~~~~~~~~~~~~

- Track input/output shapes
- Validate dimensions
- Handle dynamic shapes
- Support shape inference

3. State Management
~~~~~~~~~~~~~~~~~~~

- Support stateful operation
- Handle initial states
- Track state shapes
- Validate state dimensions

4. Time Processing
~~~~~~~~~~~~~~~~~~

- Support variable time steps
- Handle time broadcasting
- Process time deltas
- Support masking

5. Serialization
~~~~~~~~~~~~~~~~

- Support model saving
- Handle custom objects
- Proper config management
- Support model cloning

This design provides a solid foundation for implementing the Keras
version of our neural circuit policies, maintaining compatibility with
Keras conventions while adding our improved functionality.
