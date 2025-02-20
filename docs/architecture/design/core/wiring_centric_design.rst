Wiring-Centric Layer Design
===========================

Core Principle
--------------

The layer system should be built around wiring patterns, with wiring
patterns controlling both structure and behavior of layers.

Wiring-Layer Integration
------------------------

1. Wiring as Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class WiringPattern:
    """Base class for wiring patterns that drive layer behavior."""
    def __init__(self, units: int):
        self.units = units
        self.layers = {}  # Layer definitions
        self.connections = {}  # Connection patterns

    def compute_layer_structure(self, input_dim: int, output_dim: int):
        """Determine layer sizes and structure."""
        raise NotImplementedError()

    def compute_connectivity(self):
        """Determine connection patterns between layers."""
        raise NotImplementedError()

2. AutoNCP Pattern
~~~~~~~~~~~~~~~~~~

.. code:: python

class AutoNCPPattern(WiringPattern):
    """Automatically configured NCP wiring pattern."""
    def __init__(self, units: int, sparsity_level: float = 0.5):
        super().__init__(units)
        self.sparsity_level = sparsity_level

    def compute_layer_structure(self, input_dim: int, output_dim: int):
        """Automatically determine layer sizes."""
        density_level = 1.0 - self.sparsity_level
        inter_and_command = self.units - output_dim
        command_size = max(int(0.4 * inter_and_command), 1)
        inter_size = inter_and_command - command_size

        return {
            'sensory': input_dim,
            'inter': inter_size,
            'command': command_size,
            'motor': output_dim
        }

    def compute_connectivity(self):
        """Determine connection patterns."""
        density = 1.0 - self.sparsity_level
        layers = self.compute_layer_structure()

        return {
            'sensory_fanout': max(int(layers['inter'] * density), 1),
            'inter_fanout': max(int(layers['command'] * density), 1),
            'command_recurrent': max(int(layers['command'] * density * 2), 1),
            'motor_fanin': max(int(layers['command'] * density), 1)
        }

3. Layer Implementation
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class WiredLayer(nn.Module):
    """Layer implementation driven by wiring pattern."""
    def __init__(self, wiring: WiringPattern):
        super().__init__()
        self.wiring = wiring
        self.structure = None
        self.connections = None

    def build(self, input_shape):
        """Build layer based on wiring pattern."""
        # Let wiring determine structure
        self.structure = self.wiring.compute_layer_structure(
            input_shape[-1],
            self.wiring.output_dim
        )

        # Let wiring determine connectivity
        self.connections = self.wiring.compute_connectivity()

        # Build layer components
        self._build_from_wiring()

Dynamic Configuration
---------------------

1. Layer Structure
~~~~~~~~~~~~~~~~~~

.. code:: python

def _build_from_wiring(self):
    """Build layer components based on wiring pattern."""
    # Create neuron groups
    self.layers = {}
    for name, size in self.structure.items():
        self.layers[name] = self._create_neuron_group(size)

    # Create connections
    self.connections = {}
    connectivity = self.wiring.compute_connectivity()
    for src, dest, pattern in connectivity.items():
        self.connections[(src, dest)] = self._create_connection(
            self.layers[src],
            self.layers[dest],
            pattern
        )

2. Connection Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def _create_connection(self, src_layer, dest_layer, pattern):
    """Create connection between layers based on pattern."""
    if pattern.type == 'dense':
        return nn.Linear(src_layer.size, dest_layer.size)
    elif pattern.type == 'sparse':
        return SparseLiquidConnection(
            src_layer.size,
            dest_layer.size,
            pattern.density
        )

Neuron Sequencing
-----------------

1. Forward Pass
~~~~~~~~~~~~~~~

.. code:: python

def forward(self, x):
    """Process input through wired layers."""
    # Let wiring determine processing order
    layer_order = self.wiring.compute_processing_order()

    # Process through layers
    activations = {'input': x}
    for layer_name in layer_order:
        layer = self.layers[layer_name]
        connections = self.get_incoming_connections(layer_name)

        # Combine inputs according to wiring
        layer_input = self.combine_inputs(
            activations,
            connections,
            self.wiring.get_combination_rule(layer_name)
        )

        # Update layer state
        activations[layer_name] = layer(layer_input)

    return activations['output']

2. State Management
~~~~~~~~~~~~~~~~~~~

.. code:: python

def update_states(self, states):
    """Update layer states according to wiring pattern."""
    new_states = {}

    # Let wiring determine update order
    update_order = self.wiring.compute_state_update_order()

    for layer_name in update_order:
        layer = self.layers[layer_name]
        current_state = states.get(layer_name)

        # Update according to wiring rules
        new_states[layer_name] = self.wiring.update_state(
            layer_name,
            current_state,
            states
        )

    return new_states

Benefits
--------

1. Wiring-Driven Architecture

- Structure determined by wiring
- Dynamic layer configuration
- Flexible connectivity patterns

2. AutoNCP Integration

- Automatic structure determination
- Density-based connectivity
- Efficient neuron utilization

3. Extensibility

- Easy to add new wiring patterns
- Flexible layer configurations
- Custom connection types

Implementation Strategy
-----------------------

1. Phase 1: Core Wiring System

- Base wiring pattern class
- Connection management
- State handling

2. Phase 2: Layer Integration

- Wiring-aware layers
- Dynamic configuration
- State management

3. Phase 3: AutoNCP Support

- Automatic structure determination
- Density-based connectivity
- Optimization features

Questions to Consider
---------------------

1. Wiring Patterns

- How to handle custom patterns?
- What validation is needed?
- How to optimize connectivity?

2. Layer Configuration

- How dynamic should configuration be?
- What parameters to expose?
- How to handle constraints?

3. Performance

- How to optimize sparse connections?
- What operations to vectorize?
- How to handle large networks?

Next Steps
----------

1. Review Current Implementation

- Study existing patterns
- Identify optimization opportunities
- Plan migration path

2. Design Core Components

- Wiring pattern interface
- Layer integration
- State management

3. Implementation Plan

- Core wiring system
- Layer adaptations
- Testing strategy
