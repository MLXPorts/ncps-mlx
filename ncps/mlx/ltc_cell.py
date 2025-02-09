# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ncps import wirings
from ncps.mini_keras import (
    layers,
    initializers,
    constraints,
    saving
)
import mlx.core as mx
from typing import Optional, Union, Dict, Tuple


@saving.register_keras_serializable(package="ncps", name="LTCCell")
class LTCCell(layers.AbstractRNNCell):
    def __init__(
        self,
        wiring,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        initialization_ranges=None,
        **kwargs
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps.
            To get a full RNN that can process sequences,
            see `ncps.tf.LTC` or wrap the cell with a `tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>`_.

        Examples::

             >>> import ncps
             >>> from ncps.tf import LTCCell
             >>>
             >>> wiring = ncps.wirings.Random(16, output_dim=2, sparsity_level=0.5)
             >>> cell = LTCCell(wiring)
             >>> rnn = tf.keras.layers.RNN(cell)
             >>> x = tf.random.uniform((1,4)) # (batch, features)
             >>> h0 = tf.zeros((1, 16))
             >>> y = cell(x,h0)
             >>>
             >>> x_seq = tf.random.uniform((1,20,4)) # (batch, time, features)
             >>> y_seq = rnn(x_seq)

        :param wiring:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param initialization_ranges:
        :param kwargs:
        """

        super().__init__(**kwargs)
        self._init_ranges = self._get_default_init_ranges()
        if initialization_ranges:
            self._validate_and_update_ranges(initialization_ranges)

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon

    def _get_default_init_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }

    def _validate_and_update_ranges(self, ranges: Dict[str, Tuple[float, float]]) -> None:
        for k, v in ranges.items():
            if k not in self._init_ranges:
                raise ValueError(
                    f"Unknown parameter '{k}' in initialization range dictionary! "
                    f"(Expected only {list(self._init_ranges.keys())})"
                )
            if k in ["gleak", "cm", "w", "sensory_w"] and v[0] < 0:
                raise ValueError(f"Initialization range of parameter '{k}' must be non-negative!")
            if v[0] > v[1]:
                raise ValueError(f"Initialization range of parameter '{k}' is not a valid range")
            self._init_ranges[k] = v

    def _get_initializer(self, param_name: str) -> initializers.Initializer:
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return initializers.Constant(minval)
        return initializers.RandomUniform(minval, maxval)

    @property
    def state_size(self):
        """Return size of cell state (number of units in wiring)"""
        try:
            return self._wiring.units
        except AttributeError:
            try:
                # Fallback to getting total number of neurons
                return self._wiring.get_neurons_of_layer(0).shape[0] + \
                       self._wiring.get_neurons_of_layer(1).shape[0] + \
                       self._wiring.get_neurons_of_layer(2).shape[0]
            except AttributeError:
                raise ValueError("The wiring object must have 'units' or 'get_neurons_of_layer' attributes")

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    def build(self, input_shape):

        # Check if input_shape is nested tuple/list
        if isinstance(input_shape[0], tuple) or isinstance(
            input_shape[0], mx.shape
        ):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        # Ensure the wiring object has a build method
        self._wiring.build(input_dim)

        param_configs = {
            "gleak": ((self.state_size,), constraints.NonNeg(), "gleak"),
            "vleak": ((self.state_size,), None, "vleak"),
            "cm": ((self.state_size,), constraints.NonNeg(), "cm"),
            "sigma": ((self.state_size, self.state_size), None, "sigma"),
            "mu": ((self.state_size, self.state_size), None, "mu"),
            "w": ((self.state_size, self.state_size), constraints.NonNeg(), "w"),
        }

        self._params = {
            name: self.add_weight(
                name=name,
                shape=shape,
                dtype=mx.float32,
                constraint=constraint,
                initializer=self._get_initializer(init_key)
            )
            for name, (shape, constraint, init_key) in param_configs.items()
        }

        self._params["erev"] = self.add_weight(
            name="erev",
            shape=(self.state_size, self.state_size),
            dtype=mx.float32,
            initializer=self.wiring.erev_initializer,
        )

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            shape=(self.sensory_size, self.state_size),
            dtype=mx.float32,
            initializer=self._get_initializer("sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            shape=(self.sensory_size, self.state_size),
            dtype=mx.float32,
            initializer=self._get_initializer("sensory_mu"),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            shape=(self.sensory_size, self.state_size),
            dtype=mx.float32,
            constraint=constraints.NonNeg(),
            initializer=self._get_initializer("sensory_w"),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            shape=(self.sensory_size, self.state_size),
            dtype=mx.float32,
            initializer=self._wiring.sensory_erev_initializer,
        )

        self._params["sparsity_mask"] = mx.array(
            mx.abs(self._wiring.adjacency_matrix), dtype=mx.float32
        )
        self._params["sensory_sparsity_mask"] = mx.array(
            mx.abs(self._wiring.sensory_adjacency_matrix), dtype=mx.float32
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                shape=(self.sensory_size,),
                dtype=mx.float32,
                initializer=initializers.Constant(1),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                shape=(self.sensory_size,),
                dtype=mx.float32,
                initializer=initializers.Constant(0),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                shape=(self.motor_size,),
                dtype=mx.float32,
                initializer=initializers.Constant(1),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                shape=(self.motor_size,),
                dtype=mx.float32,
                initializer=initializers.Constant(0),
            )
        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = mx.expand_dims(v_pre, axis=-1)  # ✅ Keras version
        mues = v_pre - mu
        x = sigma * mues
        return mx.sigmoid(x)  # ✅ Keras sigmoid

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # Pre-compute activations using mini_keras operations
        sensory_acts = layers.activation.sigmoid(
            (mx.expand_dims(inputs, -1) - self._params["sensory_mu"]) 
            * self._params["sensory_sigma"]
        )
        sensory_w_activation = (
            self._params["sensory_w"] 
            * sensory_acts 
            * self._params["sensory_sparsity_mask"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = mx.sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant
        cm_t = self._params["cm"] / mx.cast(
            elapsed_time / self._ode_unfolds, dtype=mx.float32
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation *= self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = mx.sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = mx.sum(w_activation, axis=1) + w_denominator_sensory

            numerator = mx.add(
               mx.add(mx.multiply(cm_t, v_pre), mx.multiply(self._params["gleak"], self._params["vleak"])),
                w_numerator
            )
            denominator = mx.add(mx.add(cm_t, self._params["gleak"]), w_denominator)

            # Avoid dividing by 0
            v_pre = mx.divide(numerator, mx.maximum(denominator, self._epsilon))
        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = mx.multiply(inputs, self._params["input_w"])
        if self._input_mapping == "affine":
            inputs = mx.add(inputs, self._params["input_b"])
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]

        if self._output_mapping in ["affine", "linear"]:
            output = mx.multiply(output, self._params["output_w"])
        if self._output_mapping == "affine":
            output = mx.add(output, self._params["output_b"])
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            elapsed_time = 1.0
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states[0], elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, [next_state]

    def get_config(self):
        config = {
            "wiring": self._wiring.get_config(),  # Use get_config() directly
            "input_mapping": self._input_mapping,
            "output_mapping": self._output_mapping,
            "ode_unfolds": self._ode_unfolds,
            "epsilon": self._epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod 
    def from_config(cls, config, custom_objects=None):
        # Extract wiring config
        wiring_config = config.pop("wiring")
        
        # Import here to avoid circular imports
        from ncps import wirings
        wiring = wirings.Wiring.from_config(wiring_config)
        
        # Create new instance with reconstructed wiring
        return cls(wiring=wiring, **config)
