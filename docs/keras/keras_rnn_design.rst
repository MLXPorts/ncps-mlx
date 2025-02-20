Keras RNN Implementations Design
================================

CfC Implementation
------------------

Overview
~~~~~~~~

The CfC class provides a high-level Closed-form Continuous-time RNN
implementation, wrapping the CfCCell with sequence processing
capabilities.

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

@keras.saving.register_keras_serializable(package="ncps")
class CfC(LiquidRNN):
    def __init__(
        self,
        wiring,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        return_sequences: bool = True,
        return_state: bool = False,
        bidirectional: bool = False,
        merge_mode: str = "concat",
        **kwargs
    ):
        """Initialize CfC RNN.

        Args:
            wiring: Neural circuit wiring
            mode: Operation mode ('default', 'pure', 'no_gate')
            activation: Activation function
            backbone_units: Backbone layer sizes
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
            return_sequences: Whether to return full sequence
            return_state: Whether to return final state
            bidirectional: Whether to process in both directions
            merge_mode: How to merge bidirectional outputs
        """
        cell = CfCCell(
            wiring=wiring,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
            **kwargs
        )

Key Components
~~~~~~~~~~~~~~

1. Configuration

.. code:: python

def get_config(self):
    """Get configuration for serialization."""
    config = super().get_config()
    cell_config = self.cell.get_config()

    # Extract relevant cell config
    config.update({
        'mode': cell_config['mode'],
        'activation': cell_config['activation'],
        'backbone_units': cell_config['backbone_units'],
        'backbone_layers': cell_config['backbone_layers'],
        'backbone_dropout': cell_config['backbone_dropout'],
        'wiring': cell_config['wiring'],
    })
    return config

@classmethod
def from_config(cls, config, custom_objects=None):
    """Create from configuration."""
    # Extract wiring configuration
    wiring_config = config.pop('wiring')
    wiring_class = getattr(ncps.wirings, wiring_config['class_name'])
    wiring = wiring_class.from_config(wiring_config['config'])

    return cls(wiring=wiring, **config)

LTC Implementation
------------------

.. _overview-1:

Overview
~~~~~~~~

The LTC class provides a high-level Liquid Time-Constant RNN
implementation, wrapping the LTCCell with sequence processing
capabilities.

.. _class-structure-1:

Class Structure
~~~~~~~~~~~~~~~

.. code:: python

@keras.saving.register_keras_serializable(package="ncps")
class LTC(LiquidRNN):
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        return_sequences: bool = True,
        return_state: bool = False,
        bidirectional: bool = False,
        merge_mode: str = "concat",
        **kwargs
    ):
        """Initialize LTC RNN.

        Args:
            wiring: Neural circuit wiring
            activation: Activation function
            backbone_units: Backbone layer sizes
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
            return_sequences: Whether to return full sequence
            return_state: Whether to return final state
            bidirectional: Whether to process in both directions
            merge_mode: How to merge bidirectional outputs
        """
        cell = LTCCell(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
            **kwargs
        )

Implementation Notes
--------------------

1. Sequence Processing
~~~~~~~~~~~~~~~~~~~~~~

1. Time Step Handling

.. code:: python

def _process_time_steps(self, inputs, time_steps=None):
    """Process time steps for sequence."""
    if time_steps is None:
        return keras.backend.ones_like(inputs[:, :, 0])
    return keras.backend.cast(time_steps, self.dtype)

2. Masking Support

.. code:: python

def compute_mask(self, inputs, mask=None):
    """Compute output mask."""
    if mask is None:
        return None
    if isinstance(inputs, (list, tuple)):
        mask = mask[0]
    if not self.return_sequences:
        return None
    return mask

2. Bidirectional Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Forward Pass

.. code:: python

def _forward_pass(self, inputs, mask=None):
    """Process sequence forward."""
    return super().call(inputs, mask=mask)

2. Backward Pass

.. code:: python

def _backward_pass(self, inputs, mask=None):
    """Process sequence backward."""
    if isinstance(inputs, (list, tuple)):
        inputs = (inputs[0][:, ::-1], inputs[1][:, ::-1])
    else:
        inputs = inputs[:, ::-1]
    outputs = super().call(inputs, mask=mask)
    if isinstance(outputs, (list, tuple)):
        return (outputs[0][:, ::-1], outputs[1])
    return outputs[:, ::-1]

3. State Management
~~~~~~~~~~~~~~~~~~~

1. Initial State Creation

.. code:: python

def _get_initial_state(self, inputs):
    """Create initial state."""
    # Get batch size from inputs
    if isinstance(inputs, (list, tuple)):
        batch_size = keras.backend.shape(inputs[0])[0]
    else:
        batch_size = keras.backend.shape(inputs)[0]

    # Create zero states
    return [
        keras.backend.zeros((batch_size, self.cell.state_size))
    ]

2. State Updates

.. code:: python

def _update_states(self, states, new_states):
    """Update RNN states."""
    if self.stateful:
        updates = []
        for state, new_state in zip(states, new_states):
            updates.append(state.assign(new_state))
        self.add_update(updates)
    return new_states

4. Training Support
~~~~~~~~~~~~~~~~~~~

1. Dropout

.. code:: python

def _apply_dropout(self, inputs, training):
    """Apply input dropout during training."""
    if training and self.dropout > 0:
        inputs = keras.layers.Dropout(self.dropout)(inputs)
    return inputs

2. Recurrent Dropout

.. code:: python

def _apply_recurrent_dropout(self, states, training):
    """Apply recurrent dropout during training."""
    if training and self.recurrent_dropout > 0:
        states = [
            keras.layers.Dropout(self.recurrent_dropout)(state)
            for state in states
        ]
    return states

5. Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Caching

.. code:: python

def _cache_time_steps(self, time_steps):
    """Cache processed time steps."""
    if not hasattr(self, '_cached_time_steps'):
        self._cached_time_steps = {}
    key = keras.backend.get_uid(time_steps)
    if key not in self._cached_time_steps:
        self._cached_time_steps[key] = self._process_time_steps(time_steps)
    return self._cached_time_steps[key]

2. Memory Management

.. code:: python

def _clear_cache(self):
    """Clear cached computations."""
    if hasattr(self, '_cached_time_steps'):
        del self._cached_time_steps

This design provides detailed implementations for both CfC and LTC RNNs
in Keras, maintaining compatibility with the framework while adding our
improved functionality.
