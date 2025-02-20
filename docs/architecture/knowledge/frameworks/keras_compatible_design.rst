Keras-Compatible NCPS Design
============================

Overview
--------

NCPS will implement Keras-compatible signatures while adding AutoNCP
capabilities under the hood. This approach allows NCPS to be used as a
drop-in replacement for Keras while providing advanced neural circuit
features.

Design Principles
-----------------

1. **Keras Compatibility**

- Match Keras class signatures
- Support Keras-style layer definitions
- Maintain Keras naming conventions
- Enable drop-in replacement

2. **Enhanced Capabilities**

- Add AutoNCP features transparently
- Extend Keras concepts for circuit design
- Provide circuit-specific optimizations
- Enable automatic topology optimization

Class Structure
---------------

1. Layer API
~~~~~~~~~~~~

.. code:: python

# Matches Keras Layer signature
class Layer:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._circuit_enabled = True  # Internal AutoNCP support

    def build(self, input_shape):
        # Standard Keras build
        # + Circuit initialization
        pass

    def call(self, inputs):
        # Standard Keras forward pass
        # + Circuit optimization
        pass

# Example Dense Layer
class Dense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        # Internal circuit configuration
        self._circuit_config = self._init_circuit_config()

2. Model API
~~~~~~~~~~~~

.. code:: python

# Matches Keras Model signature
class Model:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._circuit_optimizer = None  # Internal AutoNCP support

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        # Standard Keras compile
        # + Circuit optimization setup
        self._setup_circuit_optimization(kwargs.get('circuit_config'))

    def fit(self, x, y, **kwargs):
        # Standard Keras fit
        # + Circuit adaptation during training
        pass

3. Optimizer API
~~~~~~~~~~~~~~~~

.. code:: python

# Matches Keras Optimizer signature
class Optimizer:
    def __init__(self, learning_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        # Internal circuit optimization support
        self._circuit_aware = True

    def apply_gradients(self, grads_and_vars):
        # Standard Keras gradient application
        # + Circuit-aware optimization
        pass

AutoNCP Integration
-------------------

1. Layer Extensions
~~~~~~~~~~~~~~~~~~~

.. code:: python

# Internal circuit support for layers
class CircuitAwareLayer:
    """Mixin for adding circuit capabilities to Keras layers."""

    def _init_circuit_config(self):
        return {
            'topology_optimization': True,
            'connection_adaptation': True,
            'performance_monitoring': True
        }

    def _optimize_circuit(self, inputs):
        if not self._circuit_enabled:
            return inputs
        # Perform circuit optimization
        return self._circuit_optimizer.optimize(inputs)

2. Model Extensions
~~~~~~~~~~~~~~~~~~~

.. code:: python

# Internal circuit support for models
class CircuitAwareModel:
    """Mixin for adding circuit capabilities to Keras models."""

    def _setup_circuit_optimization(self, config=None):
        self._circuit_optimizer = CircuitOptimizer(
            model=self,
            config=config or {}
        )

    def _adapt_circuit(self, inputs, outputs):
        if not self._circuit_optimizer:
            return
        self._circuit_optimizer.adapt(inputs, outputs)

Configuration System
--------------------

1. Circuit Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Can be passed to model.compile()
circuit_config = {
    'optimization': {
        'enabled': True,
        'mode': 'auto',
        'adaptation_rate': 0.01
    },
    'topology': {
        'allow_new_connections': True,
        'allow_pruning': True
    },
    'monitoring': {
        'track_performance': True,
'track_memory': True
}}}}}}}}}}}}}}}}}}}}

2. Layer Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Can be passed to layer constructor
layer_config = {
    'circuit_enabled': True,
    'optimization_mode': 'auto',
'connection_type': 'dynamic'
}}}}}}}}}}}}}}}}}}}}}}}}}}}}

Usage Examples
--------------

1. Basic Usage (Keras-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Looks exactly like Keras
model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
metrics=['accuracy']
))))))))))))))))))))

2. Circuit-Enabled Usage
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Same Keras API with circuit features
model = Sequential([
    Dense(64, activation='relu', circuit_enabled=True),
    Dense(32, activation='relu', circuit_enabled=True),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
circuit_config=circuit_config  # Optional circuit configuration
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

Implementation Strategy
-----------------------

1. **Layer Implementation**

- Start with Keras layer signatures
- Add circuit capabilities internally
- Maintain backward compatibility
- Enable gradual feature adoption

2. **Model Implementation**

- Mirror Keras model structure
- Add circuit optimization hooks
- Preserve Keras training workflow
- Enable transparent optimization

3. **Optimizer Implementation**

- Use Keras optimizer patterns
- Add circuit-aware optimization
- Support custom circuit strategies
- Maintain optimization efficiency

Migration Path
--------------

1. **For Keras Users**

.. code:: python

# Old Keras code works unchanged
from ncps import keras as keras  # Drop-in replacement

2. **For Advanced Users**

.. code:: python

# Access circuit features explicitly
from ncps.keras import Dense, CircuitConfig

Testing Strategy
----------------

1. **Compatibility Testing**

- Verify Keras API compliance
- Test drop-in replacement
- Validate signature matching
- Check backward compatibility

2. **Circuit Testing**

- Test optimization features
- Verify topology adaptation
- Validate performance gains
- Check memory efficiency
