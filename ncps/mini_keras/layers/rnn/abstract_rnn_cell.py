from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.layer import Layer
from abc import ABC
from .rnn import RNN

@keras_mini_export("ncps.mini_keras.layers.AbstractRNNCell")
class AbstractRNNCell(RNN, ABC):
    """Abstract base class for RNN cells.

    This class defines the interface for custom RNN cells. Subclasses must implement
    the `state_size`, `output_size`, and `call` methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return self.state_size

    @property
    def output_size(self):
        return self.output_size

    def build(self, input_shape):
        self.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        return self.call(inputs, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.get_initial_state(inputs, batch_size, dtype)

    def get_config(self):
        config = super().get_config()
        config.update(self.get_config())
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def reset_recurrent_dropout_mask(self):
        self.base_cell.reset_recurrent_dropout_mask()
