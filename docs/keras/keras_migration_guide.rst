Neural Circuit Policies (NCPs) Keras Migration Guide
====================================================

Overview
--------

This guide helps you migrate from the current Keras implementation to
the improved version, which offers better backbone network handling,
enhanced state management, and improved dimension tracking.

Key Changes
-----------

1. Wiring System
~~~~~~~~~~~~~~~~

Before
^^^^^^

.. code:: python

from ncps.keras import CfC

# Direct unit specification
model = CfC(units=32)

After
^^^^^

.. code:: python

from ncps.keras import CfC, FullyConnected

# Using wiring pattern
wiring = FullyConnected(units=32)
model = CfC(wiring)

2. Backbone Networks
~~~~~~~~~~~~~~~~~~~~

.. _before-1:

Before
^^^^^^

.. code:: python

# Single backbone layer
model = CfC(
    units=32,
    backbone_units=128,
backbone_layers=1
)))))))))))))))))

.. _after-1:

After
^^^^^

.. code:: python

# Multiple backbone layers
wiring = FullyConnected(units=32)
model = CfC(
    wiring,
    backbone_units=[128, 64],  # List of layer sizes
backbone_layers=2
)))))))))))))))))

3. Time-Aware Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. _before-2:

Before
^^^^^^

.. code:: python

# Time steps as additional input
inputs = [x, time_steps]
output = model(inputs)

.. _after-2:

After
^^^^^

.. code:: python

# More flexible time handling
from ncps.keras import CfC, Random

wiring = Random(units=32, sparsity_level=0.5)
model = CfC(wiring)

# Multiple ways to specify time
output1 = model(x)  # Default time step = 1.0
output2 = model([x, time_steps])  # Variable time steps
output3 = model(x, time_delta=time_steps)  # Named parameter

New Features
------------

1. Advanced Wiring Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

from ncps.keras import CfC, AutoNCP

# Automated NCP configuration
wiring = AutoNCP(
    units=32,
    output_size=10,
sparsity_level=0.5
))))))))))))))))))

model = CfC(wiring)

2. Enhanced State Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Return sequences and state
model = CfC(
    wiring,
    return_sequences=True,
return_state=True
)))))))))))))))))

# Get both output and final state
output, final_state = model(x)

# Use state for continuation
next_output, next_state = model(x2, initial_state=final_state)

3. Bidirectional Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Bidirectional with merge mode
model = CfC(
    wiring,
    bidirectional=True,
merge_mode="concat"  # or "sum", "mul", "ave"
)))))))))))))))))))))))))))))))))))))))))))))

Common Patterns
---------------

1. Basic Usage
~~~~~~~~~~~~~~

.. code:: python

from ncps.keras import CfC, FullyConnected
import keras

# Create model
wiring = FullyConnected(units=32)
rnn = CfC(wiring)

# Build sequential model
model = keras.Sequential([
    keras.layers.Input(shape=(None, input_dim)),
    rnn,
    keras.layers.Dense(output_dim)
])

2. Custom Wiring
~~~~~~~~~~~~~~~~

.. code:: python

from ncps.keras import CfC, NCP

# Create NCP wiring
wiring = NCP(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=4,
    sensory_fanout=4,
    inter_fanout=4,
    recurrent_command_synapses=3,
motor_fanin=4
)))))))))))))

# Create model
model = CfC(wiring)

3. Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Complex model with all features
model = CfC(
    wiring=AutoNCP(units=32, output_size=10),
    mode="default",
    activation="lecun_tanh",
    backbone_units=[64, 32],
    backbone_layers=2,
    backbone_dropout=0.1,
    return_sequences=True,
    return_state=True,
    bidirectional=True,
merge_mode="concat"
)))))))))))))))))))

Best Practices
--------------

1. Wiring Selection
~~~~~~~~~~~~~~~~~~~

Choose the appropriate wiring pattern based on your needs: -
FullyConnected: Dense connectivity, good baseline - Random: Controlled
sparsity, better scaling - NCP: Hierarchical structure, complex control

- AutoNCP: Automated configuration, easy scaling

2. Backbone Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Consider these factors when configuring backbones: - Layer sizes should
generally decrease - More layers for complex tasks - Add dropout for
regularization - Match final size to task needs

3. Time Handling
~~~~~~~~~~~~~~~~

Best practices for time-aware processing: - Use consistent time scales -
Normalize time steps if needed - Consider variable time steps - Handle
masked sequences properly

4. State Management
~~~~~~~~~~~~~~~~~~~

Tips for managing RNN state: - Initialize states appropriately - Clear
states between sequences - Use stateful mode carefully - Handle
bidirectional states properly

Troubleshooting
---------------

1. Dimension Mismatches
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Common fix for dimension issues
wiring = FullyConnected(
    units=32,
output_dim=desired_output_size  # Explicitly set output dimension
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

2. Memory Issues
~~~~~~~~~~~~~~~~

.. code:: python

# Reduce memory usage
wiring = Random(
    units=32,
sparsity_level=0.8  # Increase sparsity
)))))))))))))))))))))))))))))))))))))))

model = CfC(
    wiring,
    backbone_layers=1,  # Reduce layers
backbone_dropout=0.2  # Add dropout
)))))))))))))))))))))))))))))))))))

3. Performance Issues
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Optimize performance
model = CfC(
    wiring,
    backbone_units=[64],  # Simpler backbone
    backbone_layers=1,
return_sequences=False  # Only get final output if possible
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

Advanced Usage
--------------

1. Custom Cells
~~~~~~~~~~~~~~~

.. code:: python

from ncps.keras import LiquidCell

class CustomCell(LiquidCell):
    def __init__(self, wiring, **kwargs):
        super().__init__(wiring, **kwargs)
        # Custom initialization

    def call(self, inputs, states, training=None):
        # Custom processing
        return output, new_states

.. _custom-wiring-1:

2. Custom Wiring
~~~~~~~~~~~~~~~~

.. code:: python

from ncps.keras import Wiring

class CustomWiring(Wiring):
    def __init__(self, units, **kwargs):
        super().__init__(units)
        # Custom initialization

    def build(self, input_dim):
        super().build(input_dim)
        # Custom connectivity

This guide helps you transition to the improved Keras implementation
while taking advantage of new features and following best practices.
