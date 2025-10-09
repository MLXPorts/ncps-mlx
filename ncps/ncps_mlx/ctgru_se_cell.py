"""Continuous-time GRU ported from LTC-SE (Bidollahkhani et al., 2023).

Original code released under Apache-2.0 in the LTC-SE repository.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .hyperprofiles import HyperProfile, load_profile


class CTGRUSECell(nn.Module):
    def __init__(
        self,
        units: int,
        profile: Optional[HyperProfile] = None,
    ) -> None:
        super().__init__()
        self.units = units
        self._profile = profile or load_profile("ctgru_tf")
        extras = self._profile.extras

        self.M = int(extras.get("M", 8))
        self._cell_clip = float(extras.get("cell_clip", -1.0))

        self.linear_tau_r: Optional[nn.Linear] = None
        self.linear_tau_s: Optional[nn.Linear] = None
        self.linear_detect: Optional[nn.Linear] = None
        self._input_dim: Optional[int] = None

        # Pre-compute ln(tau) table and decay factors
        tau = 1.0
        ln_tau_values = []
        for _ in range(self.M):
            ln_tau_values.append(np.log(tau))
            tau = tau * (10.0 ** 0.5)
        self._ln_tau_table_np = np.array(ln_tau_values, dtype=np.float32)
        self._ln_tau_table = mx.array(self._ln_tau_table_np.reshape(1, 1, self.M))
        with np.errstate(divide="ignore"):
            decay = np.exp(-1.0 / self._ln_tau_table_np)
        self._decay = mx.array(decay.reshape(1, 1, self.M))

    def _ensure_parameters(self, input_dim: int) -> None:
        if self._input_dim == input_dim and self.linear_tau_r is not None:
            return

        fused_dim = input_dim + self.units
        self.linear_tau_r = nn.Linear(fused_dim, self.units * self.M, bias=True)
        self.linear_tau_s = nn.Linear(fused_dim, self.units * self.M, bias=True)
        self.linear_detect = nn.Linear(input_dim + self.units, self.units, bias=True)
        self._input_dim = input_dim

    def __call__(self, inputs: mx.array, state: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        self._ensure_parameters(inputs.shape[-1])

        batch_size = inputs.shape[0]
        if state is None:
            state = mx.zeros((batch_size, self.units * self.M), dtype=mx.float32)

        h_hat = mx.reshape(state, (batch_size, self.units, self.M))
        h = mx.sum(h_hat, axis=2)

        fused_input = mx.concatenate([inputs, h], axis=1)

        ln_tau_r = self.linear_tau_r(fused_input)
        ln_tau_r = mx.reshape(ln_tau_r, (batch_size, self.units, self.M))
        sf_input_r = -(ln_tau_r - self._ln_tau_table) ** 2
        rki = mx.softmax(sf_input_r, axis=2)

        q_input = mx.sum(rki * h_hat, axis=2)
        reset_value = mx.concatenate([inputs, q_input], axis=1)
        qk = mx.tanh(self.linear_detect(reset_value))
        qk = mx.reshape(qk, (batch_size, self.units, 1))

        ln_tau_s = self.linear_tau_s(fused_input)
        ln_tau_s = mx.reshape(ln_tau_s, (batch_size, self.units, self.M))
        sf_input_s = -(ln_tau_s - self._ln_tau_table) ** 2
        ski = mx.softmax(sf_input_s, axis=2)

        h_hat_next = ((1 - ski) * h_hat + ski * qk) * self._decay

        if self._cell_clip > 0:
            h_hat_next = mx.clip(h_hat_next, -self._cell_clip, self._cell_clip)

        h_next = mx.sum(h_hat_next, axis=2)
        state_next = mx.reshape(h_hat_next, (batch_size, self.units * self.M))

        return h_next, state_next
