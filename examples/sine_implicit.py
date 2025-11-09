"""MLX LTC example using implicit parameter constraints."""

from __future__ import annotations

from ncps.neurons import LTC
from ncps.wirings import FullyConnected

from mlx_common import make_sine_dataset, train_sequence_model


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=64, batch_size=2, double_frequency=True)

    wiring = FullyConnected(units=16, output_dim=1)
    model = LTC(
        input_size=inputs.shape[-1],
        units=wiring,
        return_sequences=True,
        implicit_param_constraints=True,
    )

    train_sequence_model(
        model,
        inputs,
        targets,
        epochs=250,
        learning_rate=8e-3,
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
