# Copyright 2020-2021 Mathias Lechner
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

import paddle
import paddle.nn as nn
import numpy as np
from .base import LiquidCell


class LTCCell(LiquidCell):
    def __init__(
        self,
        wiring,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs
    ):
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            **kwargs
        )
        # Store input/output mapping settings
        # Store input/output mapping settings
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._params = {}

    @property
    def sensory_size(self):
        return self.wiring.input_dim

    @property
    def motor_size(self):
        return self.wiring.output_dim

    @property
    def synapse_count(self):
        return np.sum(np.abs(self.wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.wiring.adjacency_matrix))

    def build(self, input_shape):
        """Build cell parameters."""
        # Build wiring if needed
        if isinstance(input_shape, tuple):
            input_dim = input_shape[-1]
        else:
            input_dim = input_shape.shape[-1]
            
        if not self.wiring.is_built():
            self.wiring.build(input_dim)
        
        # Set input dimension
        self._input_size = input_dim
        
        # Build backbone if needed
        self.build_backbone()
        
        # Get effective input dimension
        if self.backbone is not None:
            input_dim = self.backbone_output_dim
        else:
            input_dim = self.input_size + self.units
            
        # Initialize parameters
        self._params = {}
        self._params["gleak"] = self.create_parameter(
            shape=(self.state_size,),
            attr=nn.initializer.Uniform(low=0.001, high=1.0)
        )
        self._params["vleak"] = self.create_parameter(
            shape=(self.state_size,),
            attr=nn.initializer.Uniform(low=-0.2, high=0.2)
        )
        self._params["cm"] = self.create_parameter(
            shape=(self.state_size,),
            attr=nn.initializer.Uniform(low=0.4, high=0.6)
        )
        self._params["sigma"] = self.create_parameter(
            shape=(self.state_size, self.state_size),
            attr=nn.initializer.Uniform(low=3, high=8)
        )
        self._params["mu"] = self.create_parameter(
            shape=(self.state_size, self.state_size),
            attr=nn.initializer.Uniform(low=0.3, high=0.8)
        )
        self._params["w"] = self.create_parameter(
            shape=(self.state_size, self.state_size),
            attr=nn.initializer.Uniform(low=0.001, high=1.0)
        )
        self._params["erev"] = paddle.to_tensor(
            self.wiring.erev_initializer(),
            dtype='float32'
        )
        
        # Initialize sensory weights
        self._params["sensory_sigma"] = self.create_parameter(
            shape=(self.sensory_size, self.state_size),
            attr=nn.initializer.Uniform(low=3, high=8)
        )
        self._params["sensory_mu"] = self.create_parameter(
            shape=(self.sensory_size, self.state_size),
            attr=nn.initializer.Uniform(low=0.3, high=0.8)
        )
        self._params["sensory_w"] = self.create_parameter(
            shape=(self.sensory_size, self.state_size),
            attr=nn.initializer.Uniform(low=0.001, high=1.0)
        )
        self._params["sensory_erev"] = paddle.to_tensor(
            self.wiring.sensory_erev_initializer(),
            dtype='float32'
        )
        
        # Set sparsity masks
        self._params["sparsity_mask"] = paddle.to_tensor(
            np.abs(self.wiring.adjacency_matrix),
            dtype='float32'
        )
        self._params["sensory_sparsity_mask"] = paddle.to_tensor(
            np.abs(self.wiring.sensory_adjacency_matrix),
            dtype='float32'
        )
        
        # Initialize input/output mappings
        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.create_parameter(
                shape=(self.sensory_size,),
                attr=nn.initializer.Constant(value=1.0)
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.create_parameter(
                shape=(self.sensory_size,),
                attr=nn.initializer.Constant(value=0.0)
            )
            
        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.create_parameter(
                shape=(self.motor_size,),
                attr=nn.initializer.Constant(value=1.0)
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.create_parameter(
                shape=(self.motor_size,),
                attr=nn.initializer.Constant(value=0.0)
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = paddle.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return nn.functional.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation *= self._params["sensory_sparsity_mask"]

        sensory_rev_activation = sensory_w_activation * \
            self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = paddle.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = paddle.sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant
        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation *= self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = paddle.sum(
                rev_activation, axis=1) + w_numerator_sensory
            w_denominator = paddle.sum(
                w_activation, axis=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self._params["gleak"] * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self._params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0: self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return nn.functional.relu(w)

    def apply_weight_constraints(self):
        self._params["w"].set_value(self._clip(self._params["w"].detach()))
        self._params["sensory_w"].set_value(self._clip(self._params["sensory_w"].detach()))
        self._params["cm"].set_value(self._clip(self._params["cm"].detach()))
        self._params["gleak"].set_value(self._clip(self._params["gleak"].detach()))

    def forward(self, inputs, states):
        """Process one step with the cell.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            states: Previous state tensors
            
        Returns:
            Tuple of (output tensor, new_states)
        """
        # Get current state
        state = states[0] if isinstance(states, (list, tuple)) else states
        
        # Process input
        x = paddle.concat([inputs, state], axis=-1)
        if self.backbone is not None:
            x = self.backbone(x)
            
        # Map inputs if needed
        inputs = self._map_inputs(inputs)
        
        # Solve ODE
        next_state = self._ode_solver(inputs, state, elapsed_time=1.0)
        
        # Map outputs
        outputs = self._map_outputs(next_state)
        
        return outputs, next_state
