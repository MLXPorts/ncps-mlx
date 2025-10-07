CfC Implementation Plan
=======================

Overview
--------

Implement CfC (Closed-form Continuous-time) cell on top of BaseCell,
matching the production implementation while using Keras 3.8 patterns.

Core Implementation
-------------------

1. Class Definition
~~~~~~~~~~~~~~~~~~~

.. code:: python

# ncps/layers/cfc.py

import keras
from .base import BaseCell

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

        # Validate mode
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")

        self.mode = mode
        self.activation = keras.activations.get(activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone_fn = None

2. Build Method
~~~~~~~~~~~~~~~

.. code:: python

def build(self, input_shape):
    """Initialize weights and layers."""
    super().build(input_shape)

    # Get input dimension
    input_dim = input_shape[-1]

    # Build backbone if needed
    if self.backbone_layers > 0:
        backbone_layers = []
        for i in range(self.backbone_layers):
            backbone_layers.append(
                keras.layers.Dense(
                    self.backbone_units,
                    self.activation,
                    name=f"backbone{i}"
                )
            )
            backbone_layers.append(
                keras.layers.Dropout(self.backbone_dropout)
            )

        self.backbone_fn = keras.Sequential(backbone_layers)
        self.backbone_fn.build((None, self.units + input_dim))
        cat_shape = self.backbone_units
    else:
        cat_shape = self.units + input_dim

    # Initialize main weights
    self.ff1_kernel = self.add_weight(
        shape=(cat_shape, self.units),
        initializer="glorot_uniform",
        name="ff1_kernel"
    )
    self.ff1_bias = self.add_weight(
        shape=(self.units,),
        initializer="zeros",
        name="ff1_bias"
    )

    # Mode-specific initialization
    if self.mode == "pure":
        self._build_pure_mode()
    else:
        self._build_gated_mode(cat_shape)

3. Call Method
~~~~~~~~~~~~~~

.. code:: python

def call(self, inputs, states, training=None):
    """Process one timestep."""
    # Get current state
    state = states[0]

    # Handle time input
    if isinstance(inputs, (tuple, list)):
        inputs, t = inputs
        t = keras.ops.reshape(t, [-1, 1])
    else:
        t = 1.0

    # Combine inputs and state
    x = keras.layers.Concatenate()([inputs, state])

    # Apply backbone if present
    if self.backbone_fn is not None:
        x = self.backbone_fn(x, training=training)

    # Apply main transformation
    ff1 = keras.ops.matmul(x, self.ff1_kernel) + self.ff1_bias

    # Mode-specific processing
    if self.mode == "pure":
        new_state = self._pure_step(ff1, t)
    else:
        new_state = self._gated_step(x, ff1, t)

    return new_state, [new_state]

4. Helper Methods
~~~~~~~~~~~~~~~~~

.. code:: python

def _build_pure_mode(self):
    """Initialize pure mode weights."""
    self.w_tau = self.add_weight(
        shape=(1, self.units),
        initializer="zeros",
        name="w_tau"
    )
    self.A = self.add_weight(
        shape=(1, self.units),
        initializer="ones",
        name="A"
    )

def _build_gated_mode(self, input_dim):
    """Initialize gated mode weights."""
    self.ff2_kernel = self.add_weight(
        shape=(input_dim, self.units),
        initializer="glorot_uniform",
        name="ff2_kernel"
    )
    self.ff2_bias = self.add_weight(
        shape=(self.units,),
        initializer="zeros",
        name="ff2_bias"
    )

    self.time_a = keras.layers.Dense(self.units, name="time_a")
    self.time_b = keras.layers.Dense(self.units, name="time_b")

def _pure_step(self, ff1, t):
    """Execute pure mode step."""
    return (
        -self.A

        * keras.ops.exp(-t * (keras.ops.abs(self.w_tau) + keras.ops.abs(ff1)))
        * ff1

        + self.A
    )

def _gated_step(self, x, ff1, t):
    """Execute gated mode step."""
    ff2 = keras.ops.matmul(x, self.ff2_kernel) + self.ff2_bias
    t_a = self.time_a(x)
    t_b = self.time_b(x)
    t_interp = keras.activations.sigmoid(-t_a * t + t_b)

    if self.mode == "no_gate":
        return ff1 + t_interp * ff2
    else:
        return ff1 * (1.0 - t_interp) + t_interp * ff2

Testing Strategy
----------------

1. Basic Tests
~~~~~~~~~~~~~~

.. code:: python

def test_cfc_modes():
    """Test all CfC modes."""
    # Test default mode
    cell = CfCCell(32, mode="default")
    output, state = cell(inputs, [initial_state])
    assert output.shape == (batch_size, 32)

    # Test pure mode
    cell = CfCCell(32, mode="pure")
    output, state = cell(inputs, [initial_state])
    assert output.shape == (batch_size, 32)

    # Test no_gate mode
    cell = CfCCell(32, mode="no_gate")
    output, state = cell(inputs, [initial_state])
    assert output.shape == (batch_size, 32)

2. Backbone Tests
~~~~~~~~~~~~~~~~~

.. code:: python

def test_backbone():
    """Test backbone network."""
    cell = CfCCell(
        32,
        backbone_units=64,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    assert cell.backbone_layers == 2
    assert cell.backbone_units == 64

3. Time Handling
~~~~~~~~~~~~~~~~

.. code:: python

def test_time_input():
    """Test time input handling."""
    cell = CfCCell(32)

    # Regular input
    output1, _ = cell(inputs, [initial_state])

    # With time
    output2, _ = cell([inputs, time], [initial_state])

    assert output1.shape == output2.shape

Success Criteria
----------------

1. Functionality
~~~~~~~~~~~~~~~~

- All modes work correctly
- Backbone network functions
- Time handling works
- State management correct

2. Compatibility
~~~~~~~~~~~~~~~~

- Works with Keras 3.8 RNN
- Matches production behavior
- Supports all features

3. Code Quality
~~~~~~~~~~~~~~~

- Clean implementation
- Good test coverage
- Clear documentation

Next Steps
----------

1. Implement core class
2. Add build method
3. Add call method
4. Implement modes
5. Add tests
6. Document thoroughly
