NCP-Centric Layer Design
========================

Core Principle
--------------

The layer system should be built around neural circuit patterns, with
NCP and AutoNCP as central organizing principles rather than
afterthoughts.

Connection-Driven Design
------------------------

1. Wiring-First Layers
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class NCPLayer(nn.Module):
    """Base layer with neural circuit patterns at its core."""
    def __init__(self, wiring=None):
        super().__init__()
        self.wiring = wiring or AutoNCP()  # Default to AutoNCP
        self.connection_pattern = None

    def build(self, input_shape):
        """Let wiring pattern influence layer structure."""
        self.connection_pattern = self.wiring.compute_pattern(input_shape)
        self._build_from_pattern(self.connection_pattern)

    def _build_from_pattern(self, pattern):
        """Construct layer internals based on connection pattern."""
        self.units = pattern.output_units
        self.connections = pattern.connections
        # Layer structure follows connection pattern

2. AutoNCP Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class AutoNCPLayer(NCPLayer):
    """Layer that automatically determines its structure."""
    def __init__(self, size_hints=None):
        super().__init__(wiring=AutoNCP(size_hints))

    def build(self, input_shape):
        """Allow AutoNCP to determine layer structure."""
        pattern = self.wiring.compute_optimal_pattern(input_shape)
        self._build_from_pattern(pattern)

3. Connection Patterns
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class ConnectionPattern:
    """Neural circuit connection pattern."""
    def __init__(self):
        self.inter_neurons = []
        self.command_neurons = []
        self.motor_neurons = []
        self.connections = {}

    def influence_layer(self, layer):
        """Shape layer's behavior based on pattern."""
        layer.units = self.total_neurons()
        layer.connectivity = self.compute_connectivity()
        layer.groups = self.neuron_groups()

Layer Types
-----------

1. Dense with NCP
~~~~~~~~~~~~~~~~~

.. code:: python

class NCPDense(NCPLayer):
    """Dense layer governed by neural circuit patterns."""
    def __init__(self, wiring=None):
        super().__init__(wiring)

    def _build_from_pattern(self, pattern):
        """Build dense connections following NCP pattern."""
        self.weight_mask = pattern.compute_mask()
        self.groups = pattern.neuron_groups()

2. Recurrent with NCP
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class NCPRecurrent(NCPLayer):
    """Recurrent layer with NCP-driven connectivity."""
    def __init__(self, wiring=None):
        super().__init__(wiring)

    def _build_from_pattern(self, pattern):
        """Structure recurrent connections by NCP."""
        self.recurrent_mask = pattern.compute_recurrent_mask()
        self.group_connections = pattern.group_connectivity()

Wiring Integration
------------------

1. Pattern Application
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class WiringPattern:
    """Base class for neural circuit patterns."""
    def apply_to_layer(self, layer):
        """Configure layer based on pattern."""
        layer.units = self.compute_units()
        layer.connectivity = self.compute_connectivity()
        layer.masks = self.compute_masks()

2. AutoNCP Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class AutoNCPPattern(WiringPattern):
    """Automatically optimized neural circuit pattern."""
    def optimize_for_layer(self, layer, input_shape):
        """Determine optimal pattern for layer."""
        units = self.compute_optimal_units(input_shape)
        connectivity = self.compute_optimal_connectivity(units)
        return self.create_pattern(units, connectivity)

Layer-Pattern Interaction
-------------------------

1. Forward Pass
~~~~~~~~~~~~~~~

.. code:: python

def forward(self, x):
    """Forward pass respecting neural circuit pattern."""
    # Apply input transformation
    h = self.transform_input(x)

    # Apply connection pattern
    h = self.apply_connectivity(h)

    # Process through neuron groups
    h = self.process_groups(h)

    return h

2. State Management
~~~~~~~~~~~~~~~~~~~

.. code:: python

def manage_state(self, state):
    """State management following circuit pattern."""
    # Update states by neuron group
    for group in self.pattern.groups:
        state[group] = self.update_group_state(state[group])

    # Apply inter-group connections
    state = self.apply_group_connections(state)

    return state

Benefits
--------

1. Natural Circuit Integration

- Connection patterns drive layer behavior
- Automatic structure optimization
- Group-aware processing

2. Flexible Architecture

- Pattern-driven configuration
- Automatic optimization
- Clear neuron group separation

3. Performance

- Efficient connectivity implementation
- Optimized group processing
- Smart state management

Implementation Strategy
-----------------------

1. Phase 1: Core Pattern System

- Implement base patterns
- Create pattern-aware layers
- Develop AutoNCP integration

2. Phase 2: Layer Types

- Pattern-driven dense layers
- NCP-aware recurrent layers
- Specialized layer types

3. Phase 3: Optimization

- Pattern computation optimization
- Connection efficiency
- State management optimization

Questions to Consider
---------------------

1. Pattern Integration

- How deeply should patterns influence layers?
- What aspects should remain configurable?
- How to handle custom patterns?

2. AutoNCP

- What optimization criteria to use?
- How to balance automation and control?
- What constraints to consider?

3. Performance

- How to optimize pattern computations?
- What connection representations to use?
- How to handle large networks?

Next Steps
----------

1. Review Current Implementation

- Study existing NCP patterns
- Analyze AutoNCP behavior
- Identify optimization opportunities

2. Design Core Components

- Pattern representation
- Layer-pattern interaction
- Optimization system

3. Plan Implementation

- Core pattern system
- Layer implementations
- Testing strategy
