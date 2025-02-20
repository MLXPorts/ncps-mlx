Keras 3.8 Layer System Design
=============================

Overview
--------

Design a layer system compatible with Keras 3.8’s architecture, focusing
on CfC and LTC implementations. This uses Keras’s Layer class for cells
and layers.rnn.RNN for RNN implementations.

Core Components
---------------

1. Base Cell
~~~~~~~~~~~~

.. code:: python

import keras
from keras import Layer

class BaseCell(Layer):
    """Base cell using Keras 3.8 Layer."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        """Keras 3.8 build method."""
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        """Process one timestep."""
        raise NotImplementedError

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state for RNN processing."""
        return [keras.backend.zeros((batch_size, self.units))]

2. CfC Implementation
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

@keras.saving.register_keras_serializable(package="ncps")
class CfCCell(BaseCell):
    """Closed-form Continuous-time cell."""

    def __init__(
        self,
        units,
        mode="default",
        activation="tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.1,
        **kwargs
    ):
        super().__init__(units, **kwargs)
        self.mode = mode
        self.activation = keras.activations.get(activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

3. LTC Implementation
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

@keras.saving.register_keras_serializable(package="ncps")
class LTCCell(BaseCell):
    """Linear Time-invariant Continuous-time cell."""

    def __init__(
        self,
        units,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs
    ):
        super().__init__(units, **kwargs)
        self.activation = keras.activations.get(activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

Usage Examples
--------------

1. Basic Usage
~~~~~~~~~~~~~~

.. code:: python

# Create CfC model
model = keras.Sequential([
    keras.layers.rnn.RNN(CfCCell(32)),
    keras.layers.Dense(10)
])

# Create LTC model
model = keras.Sequential([
    keras.layers.rnn.RNN(LTCCell(32)),
    keras.layers.Dense(10)
])

2. With Custom Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# CfC with backbone
cell = CfCCell(
    units=32,
    mode="default",
    backbone_units=64,
backbone_layers=2
)))))))))))))))))

# Use in RNN
rnn = keras.layers.rnn.RNN(cell)

Implementation Details
----------------------

1. CfC Cell Build
~~~~~~~~~~~~~~~~~

.. code:: python

def build(self, input_shape):
    """Build CfC cell weights."""
    # Input processing
    input_dim = input_shape[-1]

    # Main transformation weights
    self.ff1_kernel = self.add_weight(
        shape=(input_dim, self.units),
        initializer="glorot_uniform",
        name="ff1_kernel"
    )
    self.ff1_bias = self.add_weight(
        shape=(self.units,),
        initializer="zeros",
        name="ff1_bias"
    )

    # Mode-specific weights
    if self.mode == "pure":
        self._build_pure_mode()
    else:
        self._build_gated_mode(input_dim)

2. LTC Cell Build
~~~~~~~~~~~~~~~~~

.. code:: python

def build(self, input_shape):
    """Build LTC cell weights."""
    # Input processing
    input_dim = input_shape[-1]

    # Main weights
    self.kernel = self.add_weight(
        shape=(input_dim, self.units),
        initializer="glorot_uniform",
        name="kernel"
    )
    self.bias = self.add_weight(
        shape=(self.units,),
        initializer="zeros",
        name="bias"
    )

    # Time constant network
    self.tau_kernel = keras.layers.Dense(
        self.units,
        name="tau_kernel"
    )

Testing Strategy
----------------

1. Basic Tests
~~~~~~~~~~~~~~

.. code:: python

def test_cell_creation():
    """Test basic cell creation."""
    cell = CfCCell(32)
    assert cell.units == 32
    assert cell.state_size == 32

2. Integration Tests
~~~~~~~~~~~~~~~~~~~~

.. code:: python

def test_keras_integration():
    """Test integration with Keras 3.8."""
    model = keras.Sequential([
        keras.layers.rnn.RNN(CfCCell(32))
    ])

    # Should work with standard Keras
    model.compile(optimizer="adam", loss="mse")

Next Steps
----------

1. Implement BaseCell

- Core Layer functionality
- State management
- Initial state handling

2. Implement CfC

- Port production code
- Adapt to Keras 3.8
- Add all modes

3. Implement LTC

- Port production code
- Adapt to Keras 3.8
- Add backbone support
