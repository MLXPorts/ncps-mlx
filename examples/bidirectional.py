"""Bidirectional processing with two MLX CfC cells."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ncps.neurons import CfC
from ncps.wirings import AutoNCP

from mlx_common import make_sine_dataset, train_sequence_model


class BidirectionalCfC(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_dim: int) -> None:
        super().__init__()
        wiring = AutoNCP(units=hidden_units, output_size=output_dim, sparsity_level=0.4)
        self.forward_cell = CfC(input_size, wiring, return_sequences=True)
        self.backward_cell = CfC(input_size, wiring, return_sequences=True)
        self.proj = nn.Linear(2 * output_dim, output_dim)

    def __call__(self, inputs: mx.array):
        forward_out, _ = self.forward_cell(inputs)
        seq_len = inputs.shape[1]
        indices = mx.arange(seq_len - 1, -1, -1, dtype=mx.int32)
        reversed_inputs = mx.take(inputs, indices, axis=1)
        backward_out, _ = self.backward_cell(reversed_inputs)
        backward_out = mx.take(backward_out, indices, axis=1)
        fused = mx.concatenate([forward_out, backward_out], axis=-1)
        return self.proj(fused), None


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=64, batch_size=4, double_frequency=True)
    model = BidirectionalCfC(input_size=inputs.shape[-1], hidden_units=24, output_dim=1)
    train_sequence_model(model, inputs, targets, epochs=200, learning_rate=5e-3)
    preds, _ = model(inputs)
    sample = mx.slice(
        preds,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 5, 1),
    )
    sample = mx.reshape(sample, (5,))
    print("bidirectional sample:", sample.tolist())


if __name__ == "__main__":
    main()
