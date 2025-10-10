"""MLX CfC example: learn a double-frequency sine."""

from __future__ import annotations

import mlx.core as mx
from ncps.mlx import CfC
from ncps.wirings import AutoNCP

from mlx_common import make_sine_dataset, train_sequence_model


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=64, batch_size=4, double_frequency=True)

    wiring = AutoNCP(units=32, output_size=1, sparsity_level=0.4)
    model = CfC(
        input_size=inputs.shape[-1],
        units=wiring,
        return_sequences=True,
        activation="lecun_tanh",
    )

    mx.eval(model.parameters())

    train_sequence_model(model, inputs, targets, epochs=300, learning_rate=5e-3)

    preds, _ = model(inputs)
    sample = mx.slice(
        preds,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 5, 1),
    )
    sample = mx.reshape(sample, (5,))
    print("sample prediction:", sample.tolist())


if __name__ == "__main__":
    main()
