MLX Neural Circuit Policies
===========================

The MLX backend provides efficient implementations of liquid neural networks using Apple's MLX framework, optimized for Apple Silicon. The implementation follows a modular design with wiring-based architecture for maximum flexibility and performance.

Base Classes
------------

LiquidCell
~~~~~~~~~~
.. py:class:: LiquidCell

    Base class for liquid neuron cells. Provides the foundational interface and shared functionality for implementing liquid neuron cells with wiring support.

    .. py:method:: __init__(wiring, activation: str = "tanh", backbone_units: Optional[List[int]] = None, backbone_layers: int = 0, backbone_dropout: float = 0.0, initializer: Optional[InitializerCallable] = None)

        Initialize the liquid cell.

        :param wiring: Neural wiring pattern instance
        :param activation: Name of activation function
        :param backbone_units: List of units in backbone layers
        :param backbone_layers: Number of backbone layers
        :param backbone_dropout: Dropout rate for backbone
        :param initializer: Weight initializer function

    .. py:method:: build_backbone(

        Build backbone network layers.

    .. py:method:: apply_backbone(x: mx.array) -> mx.array

        Apply backbone network if present.

        :param x: Input tensor
        :return: Processed tensor

    .. py:method:: __call__(x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array

        Process one step with the cell.

        :param x: Input tensor of shape [batch_size, input_size
        :param state: Previous state tensor of shape [batch_size, hidden_size
        :param time: Time delta since last update
        :return: Tuple of (output, new_state) tensors

    .. py:method:: get_config() -> Dict[str, Any

        Get configuration for initialization.

        :return: Configuration dictionary

    .. py:method:: state_dict() -> Dict[str, Any

        Get serializable state.

        :return: State dictionary

    .. py:method:: load_state_dict(state_dict: Dict[str, Any -> None

        Load cell state from dictionary.

        :param state_dict: State dictionary to load

LiquidRNN
~~~~~~~~~
.. py:class:: LiquidRNN

    Base class for liquid neural networks. Provides sequence processing capabilities with support for bidirectional operation and time-aware updates.

    .. py:method:: __init__(cell: LiquidCell, return_sequences: bool = False, return_state: bool = False, bidirectional: bool = False, merge_mode: Optional[str] = None

        Initialize the liquid neural network.

        :param cell: The recurrent cell instance
        :param return_sequences: Whether to return full sequence
        :param return_state: Whether to return final state
        :param bidirectional: Whether to process bidirectionally
        :param merge_mode: How to merge bidirectional outputs ("concat", "sum", "mul", "ave"

    .. py:method:: __call__(inputs: mx.array, initial_states: Optional[List[mx.array = None, time_delta: Optional[Union[float, mx.array = None) -> Union[mx.array, Tuple[mx.array, List[mx.array

        Process sequence of inputs.

        :param inputs: Input tensor of shape [batch_size, seq_len, input_size
        :param initial_states: Optional list of initial states
        :param time_delta: Optional time steps between sequence elements
        :return: Outputs or tuple of (outputs, states

    .. py:method:: state_dict() -> Dict[str, Any

        Get configuration for serialization.

        :return: State dictionary

    .. py:method:: load_state_dict(state_dict: Dict[str, Any -> None

        Load state from dictionary.

        :param state_dict: State dictionary to load

Implementations
---------------

CfCCell
~~~~~~~
.. py:class:: CfCCell

    A Closed-form Continuous-time (CfC) cell implementation.

    .. py:method:: __init__(wiring, mode: str = "pure", activation: str = "tanh", backbone_units: Optional[List[int = None, backbone_layers: int = 0, backbone_dropout: float = 0.0, initializer: Optional[InitializerCallable] = None

        Initialize CfC cell.

        :param wiring: Neural wiring pattern
        :param mode: Operation mode ("pure" or "no_gate"
        :param activation: Activation function name
        :param backbone_units: Units in backbone layers
        :param backbone_layers: Number of backbone layers
        :param backbone_dropout: Dropout rate for backbone
        :param initializer: Weight initializer function

LTCCell
~~~~~~~
.. py:class:: LTCCell

    A Liquid Time-Constant (LTC) cell implementation.

    .. py:method:: __init__(wiring, activation: str = "tanh", backbone_units: Optional[List[int = None, backbone_layers: int = 0, backbone_dropout: float = 0.0, initializer: Optional[InitializerCallable] = None

        Initialize LTC cell.

        :param wiring: Neural wiring pattern
        :param activation: Activation function name
        :param backbone_units: Units in backbone layers
        :param backbone_layers: Number of backbone layers
        :param backbone_dropout: Dropout rate for backbone
        :param initializer: Weight initializer function

CfC
~~~
.. py:class:: CfC

    A Closed-form Continuous-time (CfC) RNN implementation.

    .. py:method:: __init__(cell: CfCCell, return_sequences: bool = False, return_state: bool = False, bidirectional: bool = False, merge_mode: Optional[str] = None

        Initialize CfC network.

        :param cell: CfC cell instance
        :param return_sequences: Whether to return full sequence
        :param return_state: Whether to return final state
        :param bidirectional: Whether to process bidirectionally
        :param merge_mode: How to merge bidirectional outputs

LTC
~~~
.. py:class:: LTC

    A Liquid Time-Constant (LTC) RNN implementation.

    .. py:method:: __init__(cell: LTCCell, return_sequences: bool = False, return_state: bool = False, bidirectional: bool = False, merge_mode: Optional[str] = None

        Initialize LTC network.

        :param cell: LTC cell instance
        :param return_sequences: Whether to return full sequence
        :param return_state: Whether to return final state
        :param bidirectional: Whether to process bidirectionally
        :param merge_mode: How to merge bidirectional outputs

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

import mlx.core as mx
import mlx.nn as nn
from ncps.mlx import CfC, CfCCell
from ncps.wirings import AutoNCP

# Create wiring
wiring = AutoNCP(

# Create CfC model
model = CfC(
cell=CfCCell(
    wiring=wiring,
        activation="tanh",
        backbone_units=[64, 64],
    backbone_layers=2
        ),
            return_sequences=True,
                bidirectional=True,
            merge_mode="concat"

            # Process sequence
            x = mx.random.normal(
            time_delta = mx.ones(
            outputs, states = model(

        Time-Aware Processing
        ~~~~~~~~~~~~~~~~~~~~~

        .. code-block:: python

        # Variable time steps
        time_delta = mx.random.uniform(
            low=0.5,
                high=1.5,
                shape=(

            # Process with time information
            outputs, states = model(

        State Management
        ~~~~~~~~~~~~~~~~

        .. code-block:: python

        # Initialize states
        batch_size = 32
        initial_state = mx.zeros(

    # Process with explicit state
    outputs, final_state = model(

Advanced Features
~~~~~~~~~~~~~~~~~

See the example notebooks for more advanced usage patterns:
pass

- examples/notebooks/mlx_cfc_example.ipynb
- examples/notebooks/mlx_ltc_rnn_example.ipynb
- examples/notebooks/mlx_advanced_profiling_guide.ipynb
- examples/notebooks/mlx_hardware_optimization.ipynb

