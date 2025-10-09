"""Continuous-time RNN cell implemented with MLX."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .liquid_utils import get_activation


class CTRNNCell(nn.Module):
    def __init__(
        self,
        units: int,
        activation: str = "tanh",
        cell_clip: Optional[float] = None,
        epsilon: float = 1e-8,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.units = units
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.cell_clip = cell_clip
        self.epsilon = epsilon

        self.input_linear: Optional[nn.Linear] = None
        self.recurrent_linear: Optional[nn.Linear] = None
        self.tau: Optional[mx.array] = None
        self._input_dim: Optional[int] = None

        if input_dim is not None:
            self._ensure_parameters(input_dim)

    def _ensure_parameters(self, input_dim: int) -> None:
        if self.input_linear is None or self.input_linear.weight.shape[1] != input_dim:
            self.input_linear = nn.Linear(input_dim, self.units, bias=False)
            scale = mx.sqrt(2.0 / (input_dim + self.units))
            self.input_linear.weight = scale * mx.random.normal(self.input_linear.weight.shape)

        if self.recurrent_linear is None or self.recurrent_linear.weight.shape[1] != self.units:
            self.recurrent_linear = nn.Linear(self.units, self.units, bias=True)
            scale = mx.sqrt(2.0 / (2 * self.units))
            self.recurrent_linear.weight = scale * mx.random.normal(self.recurrent_linear.weight.shape)
            self.recurrent_linear.bias = 0.1 * mx.ones((self.units,))

        if self.tau is None or self.tau.shape[-1] != self.units:
            self.tau = 0.1 * mx.ones((self.units,))

        self._input_dim = input_dim

    def __call__(self, inputs: mx.array, state: mx.array, time: float = 1.0) -> tuple[mx.array, mx.array]:
        self._ensure_parameters(inputs.shape[-1])

        net = self.input_linear(inputs) + self.recurrent_linear(state)
        target = self.activation(net)
        tau = self.tau + self.epsilon
        d_state = (-state + target) / tau
        output = state + time * d_state
        if self.cell_clip is not None:
            output = mx.clip(output, -self.cell_clip, self.cell_clip)
        return output, output
