"""Inspect hidden states from an MLX CfC."""

from __future__ import annotations

import mlx.core as mx

from ncps.neurons import CfC
from ncps.wirings import AutoNCP

from mlx_common import make_sine_dataset


def main() -> None:
    inputs, _ = make_sine_dataset(seq_len=32, batch_size=1, double_frequency=False)

    wiring = AutoNCP(units=20, output_size=1, sparsity_level=0.4)
    model = CfC(
        input_size=inputs.shape[-1],
        units=wiring,
        return_sequences=True,
    )

    mx.eval(model.parameters())

    outputs, states = model(inputs)
    print("outputs shape:", outputs.shape)
    print("final state shape:", states.shape)
    first_step = mx.slice(
        outputs,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 1, 1),
    )
    first_step = mx.reshape(first_step, ())
    first_hidden = mx.slice(
        states,
        mx.array([0, 0], dtype=mx.int32),
        (0, 1),
        (1, 5),
    )
    first_hidden = mx.reshape(first_hidden, (5,))
    print("first output step:", float(first_step.item()))
    print("first five hidden activations:", first_hidden.tolist())


if __name__ == "__main__":
    main()
