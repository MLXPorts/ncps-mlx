NCPS Layer System Design
========================

Overview
--------

Create a layer system with identical signatures to Keras 3.8, focused
solely on supporting CfC and LTC neurons. This allows our layers to be
used as drop-in replacements while providing neural circuit
capabilities.

Core Layer System
-----------------

Base Layer
~~~~~~~~~~

.. code:: python

class Layer:
    """Base layer matching Keras Layer signature."""

    def __init__(self, **kwargs):
        """Match Keras Layer.__init__ signature exactly."""
        self.built = False
        self.trainable = True
        # ... other Keras standard attributes

    def build(self, input_shape):
        """Match Keras build signature."""
        self.built = True

    def call(self, inputs, training=None):
        """Match Keras call signature."""
        raise NotImplementedError

    def get_config(self):
        """Match Keras get_config signature."""
        return {}

RNN Cell Interface
~~~~~~~~~~~~~~~~~~

.. code:: python

class RNNCell(Layer):
    """Base RNN cell matching Keras RNNCell signature."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def state_size(self):
        """Match Keras state_size property."""
        raise NotImplementedError

    def call(self, inputs, states, training=None):
        """Match Keras RNNCell call signature."""
        raise NotImplementedError

Core Implementations
--------------------

1. CfC Cell
~~~~~~~~~~~

.. code:: python

class CfCCell(RNNCell):
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
        """Match signature from production CfC."""
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # Match production CfC build exactly
        pass

    def call(self, inputs, states, training=None):
        # Match production CfC call exactly
        pass

2. LTC Cell
~~~~~~~~~~~

.. code:: python

class LTCCell(RNNCell):
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
        """Match signature from production LTC."""
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # Match production LTC build exactly
        pass

    def call(self, inputs, states, training=None):
        # Match production LTC call exactly
        pass

Usage Examples
--------------

1. Direct Usage
~~~~~~~~~~~~~~~

.. code:: python

# Create CfC cell
cell = CfCCell(
    units=32,
    mode="default",
activation="tanh"
)))))))))))))))))

# Use in RNN layer
rnn = RNN(cell)

2. Model Integration
~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Works exactly like Keras
model = Sequential([
    RNN(CfCCell(32)),
    Dense(10)
])

# Standard Keras compilation
model.compile(
    optimizer='adam',
loss='mse'
))))))))))

Implementation Plan
-------------------

Phase 1: Core Layer System
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Implement Layer base class
2. Implement RNNCell interface
3. Match Keras signatures exactly

Phase 2: CfC Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Port production CfC code
2. Match signatures perfectly
3. Ensure drop-in compatibility

Phase 3: LTC Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Port production LTC code
2. Match signatures perfectly
3. Ensure drop-in compatibility

Testing Strategy
----------------

1. Signature Tests
~~~~~~~~~~~~~~~~~~

.. code:: python

def test_signatures():
    """Verify signatures match Keras exactly."""
    # Compare CfC signature
    assert signature(CfCCell) == signature(keras.layers.AbstractRNNCell)

    # Compare LTC signature
    assert signature(LTCCell) == signature(keras.layers.AbstractRNNCell)

2. Compatibility Tests
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def test_keras_compatibility():
    """Verify layers work as Keras layers."""
    # Should work in Keras model
    model = keras.Sequential([
        keras.layers.RNN(CfCCell(32))
    ])

    # Should support Keras features
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train)

Success Criteria
----------------

1. Exact Signature Match

- All method signatures match Keras 3.8
- All properties match Keras 3.8
- All defaults match Keras 3.8

2. Drop-in Compatibility

- Works in Keras models
- Supports all Keras features
- No special handling needed

3. Functionality

- CfC works exactly like production
- LTC works exactly like production
- All modes and options supported
