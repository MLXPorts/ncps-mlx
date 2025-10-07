Keras Wiring System Design
==========================

Base Wiring Class
-----------------

Overview
~~~~~~~~

The Wiring base class will provide the foundation for neural
connectivity patterns in Keras, supporting both dense and sparse
connections with proper dimension handling and serialization.

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

class Wiring(keras.layers.Layer):
    def __init__(self, units: int):
        """Initialize wiring pattern.

        Args:
            units: Number of neurons in the circuit
        """
        super().__init__()
        self.units = units
        self.adjacency_matrix = None
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None

Key Components
~~~~~~~~~~~~~~

1. Initialization

.. code:: python

def build(self, input_dim: int):
    """Build wiring pattern.

    Args:
        input_dim: Input dimension
    """
    self.input_dim = input_dim
    self.sensory_adjacency_matrix = keras.backend.zeros((input_dim, self.units))

2. Synapse Management

.. code:: python

def add_synapse(self, src: int, dest: int, polarity: int = 1):
    """Add synapse between neurons.

    Args:
        src: Source neuron index
        dest: Destination neuron index
        polarity: Synapse polarity (1 or -1)
    """
    if self.adjacency_matrix is None:
        self.adjacency_matrix = keras.backend.zeros((self.units, self.units))
    self.adjacency_matrix[src, dest].assign(float(polarity))

3. Dimension Handling

.. code:: python

@property
def output_size(self) -> int:
    """Get output dimension."""
    return self.output_dim or self.units

@property
def state_size(self) -> int:
    """Get state dimension."""
    return self.units

Interface Methods
~~~~~~~~~~~~~~~~~

1. Configuration

.. code:: python

def get_config(self) -> dict:
    """Get configuration for serialization."""
    return {
        'units': self.units,
        'adjacency_matrix': keras.backend.get_value(self.adjacency_matrix),
        'sensory_adjacency_matrix': keras.backend.get_value(self.sensory_adjacency_matrix),
        'input_dim': self.input_dim,
        'output_dim': self.output_dim,
    }

@classmethod
def from_config(cls, config: dict) -> 'Wiring':
    """Create from configuration."""
    instance = cls(config['units'])
    if config['adjacency_matrix'] is not None:
        instance.adjacency_matrix = keras.backend.constant(config['adjacency_matrix'])
    if config['sensory_adjacency_matrix'] is not None:
        instance.sensory_adjacency_matrix = keras.backend.constant(config['sensory_adjacency_matrix'])
    instance.input_dim = config['input_dim']
    instance.output_dim = config['output_dim']
    return instance

Specialized Wiring Patterns
---------------------------

1. FullyConnected
~~~~~~~~~~~~~~~~~

.. code:: python

class FullyConnected(Wiring):
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        self_connections: bool = True
    ):
        """Initialize fully connected wiring.

        Args:
            units: Number of neurons
            output_dim: Output dimension (default: units)
            self_connections: Allow self connections (default: True)
        """
        super().__init__(units)
        self.output_dim = output_dim or units
        self.self_connections = self_connections

2. Random
~~~~~~~~~

.. code:: python

class Random(Wiring):
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        sparsity_level: float = 0.5,
        random_seed: int = None
    ):
        """Initialize random sparse wiring.

        Args:
            units: Number of neurons
            output_dim: Output dimension (default: units)
            sparsity_level: Connection sparsity (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        super().__init__(units)
        self.output_dim = output_dim or units
        self.sparsity_level = sparsity_level
        self.random_seed = random_seed

3. NCP (Neural Circuit Policy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class NCP(Wiring):
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int,
        inter_fanout: int,
        recurrent_command_synapses: int,
        motor_fanin: int,
        seed: int = None
    ):
        """Initialize NCP wiring.

        Args:
            inter_neurons: Number of interneurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_fanout: Sensory neuron fanout
            inter_fanout: Interneuron fanout
            recurrent_command_synapses: Recurrent command connections
            motor_fanin: Motor neuron fanin
            seed: Random seed for reproducibility
        """
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self.output_dim = motor_neurons

4. AutoNCP
~~~~~~~~~~

.. code:: python

class AutoNCP(NCP):
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: int = None
    ):
        """Initialize automated NCP wiring.

        Args:
            units: Total number of neurons
            output_size: Number of output neurons
            sparsity_level: Connection sparsity (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        # Calculate architecture
        density = 1.0 - sparsity_level
        remaining = units - output_size
        command = max(int(0.4 * remaining), 1)
        inter = remaining - command

        super().__init__(
            inter_neurons=inter,
            command_neurons=command,
            motor_neurons=output_size,
            sensory_fanout=max(int(inter * density), 1),
            inter_fanout=max(int(command * density), 1),
            recurrent_command_synapses=max(int(command * density * 2), 1),
            motor_fanin=max(int(command * density), 1),
            seed=seed
        )

Implementation Notes
--------------------

1. Keras Integration
~~~~~~~~~~~~~~~~~~~~

- Use Keras backend operations
- Support Keras tensor operations
- Handle Keras variable updates
- Support Keras serialization

2. Memory Management
~~~~~~~~~~~~~~~~~~~~

- Efficient sparse storage
- Lazy matrix initialization
- Proper cleanup
- Memory-efficient updates

3. Validation
~~~~~~~~~~~~~

- Input dimension checks
- Connectivity validation
- Parameter bounds
- Shape consistency

4. Performance
~~~~~~~~~~~~~~

- Efficient sparse operations
- Optimized connectivity
- Minimal memory usage
- Fast initialization

5. Extensibility
~~~~~~~~~~~~~~~~

- Easy pattern addition
- Flexible configuration
- Clear interfaces
- Proper documentation

This design provides a robust wiring system for the Keras
implementation, supporting various connectivity patterns while
maintaining efficiency and flexibility.
