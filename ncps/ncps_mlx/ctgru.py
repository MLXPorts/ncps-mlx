"""CTGRU recurrent layer using the MLX cell implementation."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .ctgru_cell import CTGRUCell


class CTGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        units: int,
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
        cell_clip: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.cell = CTGRUCell(units=units, cell_clip=cell_clip, input_dim=input_size)

    def __call__(self, inputs: mx.array, hx: Optional[mx.array] = None, timespans: Optional[mx.array] = None):
        is_batched = inputs.ndim == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        if not is_batched:
            inputs = mx.expand_dims(inputs, axis=batch_dim)
            if timespans is not None:
                timespans = mx.expand_dims(timespans, axis=batch_dim)

        batch_size = inputs.shape[batch_dim]
        seq_len = inputs.shape[seq_dim]

        if hx is None:
            h_state = mx.zeros((batch_size, self.cell.units), dtype=mx.float32)
        else:
            h_state = hx if is_batched or hx.ndim == 2 else mx.expand_dims(hx, axis=0)

        outputs = []
        for t in range(seq_len):
            step_input = inputs[:, t, :] if self.batch_first else inputs[t, :, :]
            ts = 1.0 if timespans is None else (timespans[:, t] if self.batch_first else timespans[t, :])
            h_state, _ = self.cell(step_input, h_state, time=ts)
            if self.return_sequences:
                outputs.append(h_state)

        if self.return_sequences:
            stacked = mx.stack(outputs, axis=seq_dim)
            readout = stacked
        else:
            readout = h_state

        if not is_batched:
            readout = mx.squeeze(readout, axis=batch_dim)
            h_state = mx.squeeze(h_state, axis=0)

        return readout, h_state
