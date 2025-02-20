Wiring Patterns
===============

The wiring system allows you to define the connectivity patterns between neurons in your liquid neural networks. This guide covers the available wiring patterns and how to use them.

Base Wiring
-----------

.. code-block:: python

from ncps.mlx.wirings import Wiring

The ``Wiring`` class is the base class for all wiring patterns. It defines:
pass

- Adjacency matrices for internal and sensory connections
- Methods for adding synapses
- Support for different neuron types
- Configuration serialization

Key attributes:
pass

- ``units``: Total number of neurons
- ``input_dim``: Number of input features
- ``output_dim``: Number of output features
- ``adjacency_matrix``: Internal connectivity matrix
- ``sensory_adjacency_matrix``: Input connectivity matrix

Built-in Patterns
-----------------

Fully Connected
~~~~~~~~~~~~~~~

.. code-block:: python

from ncps.mlx.wirings import FullyConnected

wiring = FullyConnected(
units=32,              # Total neurons
output_dim=10,         # Output size
self_connections=True  # Allow self-loops

Creates a network where every neuron is connected to every other neuron.

- Pros: Maximum expressivity
- Cons: Higher memory usage, may overfit
- Use when: Working with small networks where expressivity is key

Random Sparse
~~~~~~~~~~~~~

.. code-block:: python

from ncps.mlx.wirings import Random

wiring = Random(
units=32,              # Total neurons
output_dim=10,         # Output size
sparsity_level=0.5     # Connection sparsity

Creates a network with random sparse connectivity.

- Pros: Better generalization, more efficient
- Cons: May miss important connections
- Use when: Working with large networks where efficiency matters

Neural Circuit Policy (
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from ncps.mlx.wirings import NCP

wiring = NCP(
inter_neurons=16,              # Layer 2 neurons
command_neurons=8,             # Layer 3 neurons
motor_neurons=4,               # Output neurons
sensory_fanout=4,             # Input connections
inter_fanout=4,               # Layer 2->3 connections
recurrent_command_synapses=3,  # Layer 3 recurrence
motor_fanin=4                  # Output connections

Creates a structured network with distinct neuron types and layers.

- Pros: Structured connectivity, biologically inspired
- Cons: More parameters to tune
- Use when: Problem has clear hierarchical structure

Automatic NCP
~~~~~~~~~~~~~

.. code-block:: python

from ncps.mlx.wirings import AutoNCP

wiring = AutoNCP(
units=32,              # Total neurons
output_size=4,         # Output size
sparsity_level=0.5     # Overall sparsity

Simplified NCP creation with automatic architecture selection.

- Pros: Easier to use, automatic parameter selection
- Cons: Less control over architecture
- Use when: Quick prototyping or unsure about NCP parameters

Using with Models
-----------------

Wiring patterns can be used with any liquid neural network model:

.. code-block:: python

from ncps.mlx import CfC, LTC

# Create wiring
wiring = AutoNCP(

# Create model with wiring
model = CfC(
    wiring=wiring,
        activation="tanh",
        backbone_units=[64],
    backbone_layers=1

    The wiring pattern determines:

    - How neurons are connected
    - Which neurons are inputs/outputs
    - The flow of information through the network

    Custom Wiring
    -------------

    You can create custom wiring patterns by subclassing ``Wiring``:

    .. code-block:: python

    class CustomWiring(
        Wiring)::,
    )
    def __init__(
        self,
            units,
                output_dim)::,
            )
            super(
            self.set_output_dim(

        # Add custom connectivity
        for i in range(
            units)::,
        )))))))))
        for j in range(
            i + 1,
                units)::,
            )))))))))
            if some_condition(
                i,
                    j)::,
                )))))
                pass
                self.add_synapse(

            Key methods to implement:

            - ``__init__``: Initialize wiring parameters
            - ``build``: Set up input connectivity
            - ``get_config``: Serialization support

            Best Practices
            --------------

            1. **Choosing a Pattern**

            - Start with simpler patterns (
            - Move to structured patterns (
        - Consider problem structure and size

        2. **Performance**

        - Use sparse patterns for large networks
        - Monitor memory usage with dense patterns
        - Profile different patterns

        3. **Custom Patterns**

        - Extend base Wiring class
        - Implement clear connectivity rules
        - Document assumptions and constraints

        4. **Integration**

        - Build wiring before creating model
        - Verify connectivity patterns
        - Test with small networks first

        Examples
        --------

        Time Series Forecasting
        ~~~~~~~~~~~~~~~~~~~~~~~

        .. code-block:: python

        # Create structured wiring for forecasting
        wiring = NCP(
    inter_neurons=32,    # Feature processing
    command_neurons=16,  # Temporal integration
    motor_neurons=1,     # Prediction
        sensory_fanout=8,
            inter_fanout=8,
                recurrent_command_synapses=4,
            motor_fanin=8

            # Create forecasting model
            model = CfC(

        Anomaly Detection
        ~~~~~~~~~~~~~~~~~

        .. code-block:: python

        # Create sparse wiring for efficiency
        wiring = Random(
            units=64,
                output_dim=1,
            sparsity_level=0.7  # High sparsity

            # Create detection model
            model = LTC(

        Common Issues
        -------------

        1. **Memory Issues**

        - Use sparse patterns for large networks
        - Monitor memory usage during training
        - Consider gradient accumulation

        2. **Performance Issues**

        - Profile different patterns
        - Adjust sparsity levels
        - Use appropriate batch sizes

        3. **Training Issues**

        - Start with simpler patterns
        - Gradually increase complexity
        - Monitor gradient flow

        Getting Help
        ------------

        If you need assistance with wiring patterns:

        1. Check example notebooks
        2. Review pattern documentation
        3. Join community discussions
        4. File issues on GitHub

