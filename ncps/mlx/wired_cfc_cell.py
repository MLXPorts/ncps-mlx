# Copyright 2022 Mathias Lechner. All rights reserved

from .cfc_cell import CfCCell
import mlx.core as mx
import ncps
from ncps.wirings import wirings
@ncps.mini_keras.utils.register_keras_serializable(package="ncps", name="WiredCfCCell")
class WiredCfCCell(ncps.mini_keras.layers.AbstractRNNCell):
    def __init__(
        self,
        wiring,
        fully_recurrent=True,
        mode="default",
        activation="lecun_tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._wiring = wiring
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                "Unknown mode '{}', valid options are {}".format(
                    mode, str(allowed_modes)
                )
            )
        self.mode = mode
        self.fully_recurrent = fully_recurrent
        if activation == "lecun_tanh":
            activation = lecun_tanh
        self._activation = activation
        self._cfc_layers = []

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def input_size(self):
        return self._wiring.input_dim

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self._wiring.build(input_dim)
        for i in range(self._wiring.num_layers):
            layer_i_neurons = self._wiring.get_neurons_of_layer(i)
            if i == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[
                    :, layer_i_neurons
                ]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(i - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, layer_i_neurons]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            if self.fully_recurrent:
                recurrent_sparsity = np.ones(
                    (len(layer_i_neurons), len(layer_i_neurons)), dtype=np.int32
                )
            else:
                recurrent_sparsity = self._wiring.adjacency_matrix[
                    layer_i_neurons, layer_i_neurons
                ]
            cell = CfCCell(
                len(layer_i_neurons),
                input_sparsity,
                recurrent_sparsity,
                mode=self.mode,
                activation=self._activation,
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0,
            )

            cell_in_shape = (None, input_sparsity.shape[0])
            # cell.build(cell_in_shape)
            self._cfc_layers.append(cell)

        self._layer_sizes = [l.units for l in self._cfc_layers]
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = mx.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = 1.0

        states = mx.split(states[0], self._layer_sizes, axis=-1)
        assert len(states) == self._wiring.num_layers
        new_hiddens = []
        for i, layer in enumerate(self._cfc_layers):
            layer_input = (inputs, t)
            output, new_hidden = layer(layer_input, [states[i]])
            new_hiddens.append(new_hidden[0])
            inputs = output

        assert len(new_hiddens) == self._wiring.num_layers
        if self._wiring.output_dim != output.shape[-1]:
            output = output[:, 0 : self._wiring.output_dim]

        new_hiddens = mx.concat(new_hiddens, axis=-1)
        return output, new_hiddens

    def get_config(self):
        seralized = self._wiring.get_config()
        seralized["mode"] = self.mode
        seralized["activation"] = self._activation
        seralized["backbone_units"] = self.hidden_units
        seralized["backbone_layers"] = self.hidden_layers
        seralized["backbone_dropout"] = self.hidden_dropout
        return seralized

    @classmethod
    def from_config(cls, config):
        wiring = wirings.Wiring.from_config(config)
        return cls(wiring=wiring, **config)