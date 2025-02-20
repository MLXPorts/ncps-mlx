Base Cell Implementation Plan
=============================

Phase 1: Core Cell Implementation
---------------------------------

Step 1: Basic Cell Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# ncps/layers/base.py

import keras
from keras import Layer

class BaseCell(Layer):
    """Foundation for CfC and LTC implementations."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.built = False

    def build(self, input_shape):
        """Initialize base attributes."""
        super().build(input_shape)
        self.built = True

    def call(self, inputs, states, training=None):
        """Process one timestep."""
        raise NotImplementedError

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Create initial state."""
        return [keras.backend.zeros((batch_size, self.units))]

Step 2: Add State Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def compute_output_shape(self, input_shape):
    """Compute output shape."""
    return input_shape[0], self.units

def get_config(self):
    """Get layer config."""
    config = super().get_config()
    config.update({
        'units': self.units
    })
    return config

Testing Plan
------------

1. Basic Tests
~~~~~~~~~~~~~~

.. code:: python

# tests/test_base_cell.py

def test_initialization():
    """Test basic cell initialization."""
    cell = BaseCell(32)
    assert cell.units == 32
    assert cell.state_size == 32
    assert not cell.built

def test_build():
    """Test build method."""
    cell = BaseCell(32)
    cell.build((None, 64))  # Batch, input_dim
    assert cell.built

def test_initial_state():
    """Test initial state creation."""
    cell = BaseCell(32)
    states = cell.get_initial_state(batch_size=16)
    assert len(states) == 1
    assert states[0].shape == (16, 32)

2. Integration Test
~~~~~~~~~~~~~~~~~~~

.. code:: python

def test_with_rnn():
    """Test base cell works with RNN."""
    class SimpleCell(BaseCell):
        def call(self, inputs, states, training=None):
            return inputs, states

    cell = SimpleCell(32)
    rnn = keras.layers.rnn.RNN(cell)
    # Should work without errors
    rnn.build((None, 10, 64))

Implementation Steps
--------------------

Step 1: Core Setup
~~~~~~~~~~~~~~~~~~

1. Create base.py file
2. Implement BaseCell class
3. Add basic attributes
4. Add required methods

Step 2: State Management
~~~~~~~~~~~~~~~~~~~~~~~~

1. Add get_initial_state
2. Add compute_output_shape
3. Add config management
4. Add build method

Step 3: Testing
~~~~~~~~~~~~~~~

1. Create test file
2. Add basic tests
3. Add integration tests
4. Verify Keras compatibility

Validation Criteria
-------------------

1. Keras Compatibility
~~~~~~~~~~~~~~~~~~~~~~

- Works with keras.layers.rnn.RNN
- Follows Keras 3.8 conventions
- Supports standard Keras features

2. Functionality
~~~~~~~~~~~~~~~~

- Proper state management
- Correct shape handling
- Config serialization

3. Code Quality
~~~~~~~~~~~~~~~

- Clean implementation
- Good test coverage
- Clear documentation

Next Steps
----------

1. Immediate
~~~~~~~~~~~~

1. Create base.py
2. Implement BaseCell
3. Add tests
4. Verify RNN compatibility

2. Short Term
~~~~~~~~~~~~~

1. Start CfC implementation
2. Add backbone support
3. Implement modes

3. Medium Term
~~~~~~~~~~~~~~

1. Implement LTC
2. Add optimizations
3. Enhance documentation

Usage Example
-------------

.. code:: python

# Example of extending BaseCell
class CustomCell(BaseCell):
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(units, **kwargs)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weight(
            'kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform'
        )

    def call(self, inputs, states, training=None):
        h = states[0]
        output = self.activation(
            keras.ops.matmul(inputs, self.kernel)
        )
        return output, [output]

# Use with RNN
rnn = keras.layers.rnn.RNN(CustomCell(32))

Success Criteria
----------------

.. _code-quality-1:

1. Code Quality
~~~~~~~~~~~~~~~

- Clean, readable code
- Well-documented
- Properly tested

.. _functionality-1:

2. Functionality
~~~~~~~~~~~~~~~~

- Works as base for CfC/LTC
- Handles states correctly
- Manages shapes properly

3. Integration
~~~~~~~~~~~~~~

- Works with Keras RNN
- Supports training
- Serializable

Documentation
-------------

1. Class Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class BaseCell(Layer):
    """Base cell for continuous-time RNN implementations.

    This serves as the foundation for CfC and LTC cells,
    providing core RNN cell functionality compatible with
    Keras 3.8's RNN layer.

    Args:
        units: Positive integer, dimensionality of the output space.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

2. Method Documentation
~~~~~~~~~~~~~~~~~~~~~~~

\```python def get_initial_state(self, inputs=None, batch_size=None,
dtype=None): “““Create initial state for RNN processing.

::

Args:
    inputs: Optional input tensor for shape inference.
    batch_size: Optional batch size if inputs not provided.
    dtype: Optional dtype for state tensors.

Returns:
List containing initial state tensor.
"""""""""""""""""""""""""""""""""""""
