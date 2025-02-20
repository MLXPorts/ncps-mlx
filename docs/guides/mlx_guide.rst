MLX Neural Circuit Policies
===========================

Introduction
------------
This guide provides comprehensive documentation for the MLX implementation of Neural Circuit Policies (NCPs), optimized for Apple Silicon processors.

Apple Silicon Optimizations
---------------------------

Hardware Utilization
~~~~~~~~~~~~~~~~~~~~
The MLX implementation is specifically optimized for Apple Silicon processors:

1. Neural Engine:

    - Automatic utilization of the Neural Engine for supported operations
    - Optimized matrix multiplications
    - Efficient activation functions

2. Memory Management:

    - Zero-copy data transfers
    - Lazy evaluation for computation graphs
    - Efficient memory layout for Apple Silicon

3. Performance Features:

    - Automatic operation fusion
    - Hardware-specific optimizations
    - Parallel computation support

Core Architecture
-----------------

Base Classes
~~~~~~~~~~~~

LiquidCell
^^^^^^^^^^
.. py:class:: LiquidCell

    Base class for liquid neuron cells with wiring support.

    .. py:method:: __init__(wiring, activation="tanh", backbone_units=None, backbone_layers=0, backbone_dropout=0.0, initializer=None

        Initialize the liquid cell.

        :param wiring: Neural wiring pattern
        :param activation: Activation function name
        :param backbone_units: Units in backbone layers
        :param backbone_layers: Number of backbone layers
        :param backbone_dropout: Dropout rate for backbone
        :param initializer: Weight initializer function

    .. py:method:: build_backbone(

        Build backbone network layers.

    .. py:method:: apply_backbone(x: mx.array) -> mx.array

        Apply backbone network to input.

        :param x: Input tensor
        :return: Processed tensor

    .. py:method:: get_config() -> Dict[str, Any

        Get configuration dictionary.

    .. py:method:: state_dict() -> Dict[str, Any

        Get serializable state.

    .. py:method:: load_state_dict(state_dict: Dict[str, Any

        Load state from dictionary.

LiquidRNN
^^^^^^^^^
.. py:class:: LiquidRNN

    Base class for liquid neural networks.

    .. py:method:: __init__(cell: LiquidCell, return_sequences: bool = False, return_state: bool = False, bidirectional: bool = False, merge_mode: Optional[str] = None

        Initialize the network.

        :param cell: The recurrent cell instance
        :param return_sequences: Whether to return full sequence
        :param return_state: Whether to return final state
        :param bidirectional: Whether to process bidirectionally
        :param merge_mode: How to merge bidirectional outputs

Neural Wiring
-------------

The wiring system defines connectivity patterns between neurons:

1. Base Wiring
^^^^^^^^^^^^^^
.. py:class:: Wiring

    Base class for neural wiring patterns.

    Key Features.. code-block:: python

- Flexible connectivity definition
- Synapse management
- State tracking
- Configuration handling

2. Wiring Patterns
^^^^^^^^^^^^^^^^^^

AutoNCP
"""""""
Automatic Neural Circuit Policy wiring:
pass

.. code-block:: python

wiring = AutoNCP(
units=32,          # Total neurons
output_size=4,     # Output neurons
sparsity_level=0.5 # Connection sparsity

NCP
"""
Manual Neural Circuit Policy wiring:
pass

.. code-block:: python

wiring = NCP(
    inter_neurons=16,
        command_neurons=8,
            motor_neurons=4,
                sensory_fanout=4,
                    inter_fanout=4,
                        recurrent_command_synapses=3,
                    motor_fanin=4

                    Model Implementation
                    --------------------

                    1. CfC Models
                    ^^^^^^^^^^^^^

                    CfCCell
                    """""""
                    .. py:class:: CfCCell

                    Closed-form Continuous-time cell implementation.

                    Features:
                    pass

                    - Multiple operation modes
                    - Backbone network support
                    - Time-aware processing
                    - State management

                    CfC
                    """
                    .. py:class:: CfC

                    High-level CfC network implementation.

                    Features:
                    pass
                    pass

                    - Bidirectional processing
                    - Sequence handling
                    - State management
                    - Time-delta support

                    2. LTC Models
                    ^^^^^^^^^^^^^

                    LTCCell
                    """""""
                    .. py:class:: LTCCell

                    Liquid Time-Constant cell implementation.

                    Features:

                    - Biologically-inspired dynamics
                    - Backbone network support
                    - Time-aware processing
                    - State management

                    LTC
                    """
                    .. py:class:: LTC

                    High-level LTC network implementation.

                    Features:
                    pass

                    - Bidirectional processing
                    - Sequence handling
                    - State management
                    - Time-delta support

                    Performance Optimization
                    ------------------------

                    1. Memory Management
                    ^^^^^^^^^^^^^^^^^^^^

                    .. code-block:: python

                    # Use lazy evaluation
                    import mlx.core as mx

                    @mx.compile(
                    def forward_pass(
                        x,
                            training=True)::,
                        )
                        pass
                        pass
                        return model(

                    2. Computation Optimization
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

                    .. code-block:: python

                    # Enable operator fusion
                    mx.set_default_device(
                    mx.enable_operator_fusion(

                3. Batch Processing
                ^^^^^^^^^^^^^^^^^^^

                .. code-block:: python

                # Optimal batch sizes for Apple Silicon
                    BATCH_SIZES = {
                        'M1': 64,
                            'M1 Pro': 128,
                                'M1 Max': 256,
                            'M1 Ultra': 512

                            Advanced Features
                            -----------------

                            1. Custom Wiring
                            ^^^^^^^^^^^^^^^^

                            .. code-block:: python

                            class CustomWiring(
                                Wiring)::,
                            )
                            def __init__(
                                self,
                                    units,
                                        output_size)::,
                                    )
                                    pass
                                    pass
                                    super(
                                self.output_size = output_size

                                def build(
                                    self,
                                        input_dim)::,
                                    )
                                    # Custom connectivity logic
                                    pass

                                    2. State Management
                                    ^^^^^^^^^^^^^^^^^^^

                                    .. code-block:: python

                                    class StatefulModel(
                                        nn.Module)::,
                                    )
                                    def __init__(
                                        self)::,
                                    )
                                    super(
                                    self.cfc = CfC(
                                    cell=CfCCell(
                                return_state=True

                                def __call__(
                                    self,
                                        x,
                                            state=None)::,
                                        )
                                        pass
                                        return self.cfc(

                                    3. Time-Aware Processing
                                    ^^^^^^^^^^^^^^^^^^^^^^^^

                                    .. code-block:: python

                                    def process_sequence(
                                        model,
                                            x,
                                                dt)::,
                                            )
                                            pass
                                            """Process sequence with variable time steps."""
                                            return model(

                                        Best Practices
                                        --------------

                                        1. Hardware Optimization
                                        ^^^^^^^^^^^^^^^^^^^^^^^^

                                        - Use appropriate batch sizes for your device
                                        - Enable operator fusion when possible
                                        - Monitor memory usage
                                        - Profile performance bottlenecks

                                        2. Model Architecture
                                        ^^^^^^^^^^^^^^^^^^^^^

                                        - Choose appropriate wiring patterns
                                        - Use backbone networks for complex tasks
                                        - Enable bidirectional processing when needed
                                        - Consider time-aware processing

                                        3. Training
                                        ^^^^^^^^^^^

                                        - Monitor loss stability
                                        - Use gradient clipping
                                        - Validate state consistency
                                        - Profile memory usage

                                        4. Testing
                                        ^^^^^^^^^^

                                        - Write comprehensive tests
                                        - Check edge cases
                                        - Validate shapes
                                        - Test performance

                                        Troubleshooting
                                        ---------------

                                        Common Issues
                                        ^^^^^^^^^^^^^

                                        1. Memory Issues:
                                        pass

                                        - Clear unused variables
                                        - Use appropriate batch sizes
                                        - Monitor memory usage
                                        - Enable garbage collection

                                        2. Performance Issues:

                                        - Profile bottlenecks
                                        - Check batch sizes
                                        - Enable optimizations
                                        - Monitor hardware utilization

                                        3. Training Issues:

                                        - Check learning rates
                                        - Monitor gradients
                                        - Validate loss computation
                                        - Check state consistency

                                        References
                                        ----------

                                        - `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
                                        - `Neural Circuit Policies Paper <https://arxiv.org/abs/2003.06567>`_
                                        - `MLX GitHub Repository <https://github.com/ml-explore/mlx>`_
                                        - `NCP GitHub Repository <https://github.com/mlech26l/ncps>`_

