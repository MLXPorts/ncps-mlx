Keras Implementation Plan for Neural Circuit Policies
=====================================================

Phase 1: Base Classes
---------------------

1. LiquidCell Base Class
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class LiquidCell(layers.Layer):
    def __init__(
        self,
        wiring,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
    ):
        # Similar to MLX implementation but using Keras components
        pass

Key Features: - Backbone network management - Dimension tracking - State
initialization - Activation handling

2. LiquidRNN Base Class
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class LiquidRNN(layers.RNN):
    def __init__(
        self,
        cell,
        return_sequences=True,
        return_state=False,
        bidirectional=False,
        merge_mode="concat",
    ):
        # Similar to MLX implementation but using Keras layers
        pass

Key Features: - Sequence processing - Bidirectional support - Time-delta
handling - State management

Phase 2: Wiring System
----------------------

1. Base Wiring Class
~~~~~~~~~~~~~~~~~~~~

.. code:: python

class Wiring(layers.Layer):
    def __init__(self, units):
        # Similar to MLX implementation
        pass

Features: - Connectivity management - Dimension handling - Synapse
configuration - State serialization

2. Specialized Wiring Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- FullyConnected
- Random
- NCP
- AutoNCP

Phase 3: Cell Implementations
-----------------------------

1. CfCCell
~~~~~~~~~~

.. code:: python

class CfCCell(LiquidCell):
    def __init__(
        self,
        wiring,
        mode="default",
        activation="lecun_tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
    ):
        # Enhanced version of current implementation
        pass

Improvements: - Better backbone integration - Enhanced dimension
handling - Improved state management - Proper initialization

2. LTCCell
~~~~~~~~~~

.. code:: python

class LTCCell(LiquidCell):
    def __init__(
        self,
        wiring,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
    ):
        # Port from MLX implementation
        pass

Features: - Time-constant dynamics - Enhanced backbone support - Better
state handling - Proper initialization

Phase 4: RNN Implementations
----------------------------

1. CfC
~~~~~~

.. code:: python

class CfC(LiquidRNN):
    def __init__(
        self,
        wiring,
        mode="default",
        activation="lecun_tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs,
    ):
        # Enhanced version of current implementation
        pass

Improvements: - Better backbone handling - Enhanced state management -
Improved dimension tracking - Proper serialization

2. LTC
~~~~~~

.. code:: python

class LTC(LiquidRNN):
    def __init__(
        self,
        wiring,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs,
    ):
        # Port from MLX implementation
        pass

Features: - Time-constant dynamics - Enhanced backbone support - Better
state handling - Proper serialization

Implementation Strategy
-----------------------

1. Base Infrastructure
~~~~~~~~~~~~~~~~~~~~~~

1. Create base classes
2. Implement wiring system
3. Add dimension tracking
4. Set up state management

2. Cell Implementation
~~~~~~~~~~~~~~~~~~~~~~

1. Update CfCCell
2. Port LTCCell
3. Add backbone improvements
4. Enhance state handling

3. RNN Implementation
~~~~~~~~~~~~~~~~~~~~~

1. Update CfC
2. Port LTC
3. Add bidirectional support
4. Improve serialization

4. Testing
~~~~~~~~~~

1. Port MLX tests
2. Add Keras-specific tests
3. Test serialization
4. Verify compatibility

Key Improvements
----------------

1. Architecture
~~~~~~~~~~~~~~~

- Better separation of concerns
- Enhanced modularity
- Improved extensibility
- Proper inheritance

2. Functionality
~~~~~~~~~~~~~~~~

- Robust backbone support
- Better time handling
- Enhanced state management
- Proper validation

3. Integration
~~~~~~~~~~~~~~

- Keras-specific optimizations
- Better serialization
- Enhanced compatibility
- Proper documentation

.. _testing-1:

4. Testing
~~~~~~~~~~

- Comprehensive test suite
- Better coverage
- Enhanced validation
- Proper isolation

Migration Guide
---------------

1. For Existing Users
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Old way
from ncps.keras import CfC
model = CfC(units=32)

# New way
from ncps.keras import CfC, FullyConnected
wiring = FullyConnected(units=32)
model = CfC(wiring)

2. For New Features
~~~~~~~~~~~~~~~~~~~

.. code:: python

# Using wiring patterns
from ncps.keras import AutoNCP, CfC

wiring = AutoNCP(
    units=32,
    output_size=10,
sparsity_level=0.5
))))))))))))))))))
model = CfC(
    wiring,
    backbone_units=[64, 32],
backbone_layers=2
)))))))))))))))))

This plan provides a structured approach to implementing the MLX
improvements in the Keras version while maintaining compatibility and
adding new features.
