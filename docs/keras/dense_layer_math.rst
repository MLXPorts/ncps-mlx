Dense Layer Mathematical Operations: Keras vs MLX
=================================================

Core Mathematical Operations
----------------------------

Both Keras and MLX implement the same fundamental Dense layer operation:

::

output = activation(dot(input, kernel) + bias)

Keras Implementation
~~~~~~~~~~~~~~~~~~~~

.. code:: python

class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True):
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel',
            shape=(input_shape[-1], self.units)
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(self.units,)
            )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

MLX Implementation
~~~~~~~~~~~~~~~~~~

.. code:: python

class Linear(nn.Module):
    def __init__(self, input_dims, output_dims, bias=True):
        super().__init__()
        self.weight = mx.random.normal((input_dims, output_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x):
        y = x @ self.weight
        if hasattr(self, 'bias'):
            y = y + self.bias
        return y

Key Differences
---------------

1. Weight Management
~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Keras - Deferred weight creation
def build(self, input_shape):
    self.kernel = self.add_weight(...)

# MLX - Immediate weight creation
def __init__(self):
    self.weight = mx.random.normal(...)

2. Matrix Operations
~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Keras
outputs = tf.matmul(inputs, self.kernel)

# MLX
y = x @ self.weight  # Direct operator overloading

3. Activation Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Keras - Built into Dense layer
if self.activation:
    outputs = self.activation(outputs)

# MLX - Separate activation layers
class Sequential:
    layers = [
        Linear(...),
        Activation(...)
    ]

Mathematical Equivalence
------------------------

Both implementations compute the same mathematical operation:

1. Matrix Multiplication:

::

    Z = XW
    where:

    - X is input of shape (batch_size, input_dims)
    - W is weights of shape (input_dims, output_dims)
    - Z is result of shape (batch_size, output_dims)

2. Bias Addition:

::

    Y = Z + b
    where:

    - b is broadcast to match Z's shape

3. Activation:

::

    output = f(Y)
    where f is the activation function

Implementation for Wiring Patterns
----------------------------------

Base Structure
~~~~~~~~~~~~~~

.. code:: python

class WiredDense(nn.Module):
    def __init__(self, input_dims, output_dims, wiring=None):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.wiring = wiring

        # Initialize weights based on wiring pattern
        if wiring:
            self.weight = self._init_wired_weights()
        else:
            self.weight = mx.random.normal((input_dims, output_dims))

Core Operations
~~~~~~~~~~~~~~~

.. code:: python

def __call__(self, x):
    # 1. Apply wiring pattern to weights if needed
    effective_weight = (
        self.wiring.apply(self.weight)
        if self.wiring
        else self.weight
    )

    # 2. Matrix multiplication
    y = x @ effective_weight

    # 3. Bias addition (if used)
    if hasattr(self, 'bias'):
        y = y + self.bias

    return y

Wiring Integration
~~~~~~~~~~~~~~~~~~

.. code:: python

def _init_wired_weights(self):
    """Initialize weights according to wiring pattern."""
    shape = (self.input_dims, self.output_dims)
    mask = self.wiring.get_mask(shape)
    weights = mx.random.normal(shape)
    return weights * mask

Key Considerations
------------------

1. Weight Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

- When to create weights
- How wiring affects initialization
- Memory efficiency

2. Operation Order
~~~~~~~~~~~~~~~~~~

- Matrix multiplication first
- Wiring pattern application
- Bias and activation

3. Shape Management
~~~~~~~~~~~~~~~~~~~

- Input validation
- Weight shapes
- Output shapes

Next Steps
----------

1. Implementation

- Core mathematical operations
- Wiring pattern integration
- Shape management

2. Validation

- Mathematical correctness
- Shape handling
- Memory efficiency

3. Documentation

- Mathematical operations
- Usage patterns
- Integration examples
