"""Saving and loading MLX CfC weights (port of the original Keras example)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ncps_mlx import CfC
from ncps_mlx.wirings import AutoNCP

from mlx_common import make_sine_dataset, train_sequence_model


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=48, batch_size=2, double_frequency=True)
    wiring = AutoNCP(units=18, output_size=1, sparsity_level=0.5)
    model = CfC(input_size=inputs.shape[-1], units=wiring, return_sequences=True)

    train_sequence_model(model, inputs, targets, epochs=150, learning_rate=5e-3)

    weights_path = Path("cfc_weights.npz")
    model.save_weights(str(weights_path))
    print(f"saved weights to {weights_path.resolve()}")

    restored = CfC(input_size=inputs.shape[-1], units=wiring, return_sequences=True)
    restored.load_weights(str(weights_path))
    preds, _ = restored(inputs)
    sample = mx.slice(
        preds,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 5, 1),
    )
    sample = mx.reshape(sample, (5,))
    print("restored sample:", sample.tolist())


if __name__ == "__main__":
    main()
