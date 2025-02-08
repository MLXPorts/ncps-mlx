from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.layer import Layer
from abc import ABC, abstractmethod

@keras_mini_export("ncps.mini_keras.layers.AbstractRNNCell")
class AbstractRNNCell(Layer, ABC):
    """Abstract base class for RNN cells.

    This class defines the interface for custom RNN cells. Subclasses must implement
    the `state_size`, `output_size`, and `call` methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_size = None
        self._output_size = None

    @property
    @abstractmethod
    def state_size(self):
        """Size of the state used by the cell."""
        return self._state_size

    @property
    @abstractmethod
    def output_size(self):
        """Size of the output produced by the cell."""
        return self._output_size

    @abstractmethod
    def build(self, input_shape):
        """Builds the cell's internal components."""
        pass

    @abstractmethod
    def call(self, inputs, states, **kwargs):
        """The main computation performed by the cell."""
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Returns the initial state for the cell."""
        if batch_size is None and inputs is not None:
            batch_size = inputs.shape[0]
        return [
            ncps.mini_keras.ops.zeros((batch_size, self.state_size), dtype=dtype)
            for _ in range(len(self.state_size) if isinstance(self.state_size, (list, tuple)) else 1)
        ]

    def get_config(self):
        """Returns the configuration of the cell."""
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
