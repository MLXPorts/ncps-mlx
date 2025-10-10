"""MLX LTC example mirroring the classic PyTorch sine tutorial."""

from __future__ import annotations

from ncps.mlx import LTC
from ncps.wirings import AutoNCP

from mlx_common import make_sine_dataset, train_sequence_model


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=64, batch_size=4, double_frequency=True)

    wiring = AutoNCP(units=24, output_size=1, sparsity_level=0.5)
    model = LTC(
        input_size=inputs.shape[-1],
        units=wiring,
        return_sequences=True,
        implicit_param_constraints=False,
    )

    train_sequence_model(
        model,
        inputs,
        targets,
        epochs=300,
        learning_rate=5e-3,
        apply_constraints=model.apply_weight_constraints,
    )

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
