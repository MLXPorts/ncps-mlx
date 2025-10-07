"""CFC regression example using MLX training patterns."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from pathlib import Path

from ncps.mlx import CfC
from ncps.mlx.wirings import FullyConnected


def make_dataset(num_sequences: int = 64, seq_len: int = 200):
    """Create synthetic sin/cos regression data."""
    base = mx.linspace(0, 4 * mx.pi, seq_len)
    # Broadcast across sequences with random phase offsets
    phases = mx.random.uniform(low=0.0, high=2 * mx.pi, shape=(num_sequences, 1))
    x = base + phases
    inputs = mx.stack(
        [mx.sin(x), mx.cos(x)],
        axis=2,
    )  # [num_sequences, seq_len, 2]
    targets = mx.sin(2.0 * x)
    targets = mx.expand_dims(targets, axis=-1)  # [num_sequences, seq_len, 1]
    return inputs, targets


def make_model(input_dim: int, hidden_dim: int, output_dim: int):
    wiring = FullyConnected(units=hidden_dim, output_dim=output_dim, erev_init_seed=42)
    wiring.build(input_dim=input_dim)
    model = CfC(
        wiring=wiring,
        return_sequences=True,
        mode="default",
        activation="lecun_tanh",
        backbone_units=[hidden_dim],
        backbone_layers=1,
        backbone_dropout=0.1,
    )
    _ = model(mx.zeros((1, 1, input_dim)))
    mx.eval(model.parameters())
    return model


def train(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    epochs: int = 200,
    checkpoint_path: Path | None = None,
):
    optimizer = optim.Adam(learning_rate=1e-3)
    # Parameters have already been materialised in make_model, but ensure we sync states.
    mx.eval(model.parameters())

    def loss_fn(mdl, xs, ys):
        preds = mdl(xs)
        return mx.mean((preds - ys) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        loss, grads = value_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch + 1:03d} | loss: {loss.tolist():.6f}")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(checkpoint_path))

    return model


def main():
    mx.random.seed(0)
    inputs, targets = make_dataset()
    model = make_model(input_dim=inputs.shape[-1], hidden_dim=32, output_dim=1)
    model.train(True)
    weights_path = Path("artifacts") / "mlx_cfc_regression_weights.npz"
    train(model, inputs, targets, checkpoint_path=weights_path)

    # Demonstrate loading the saved weights into a fresh model instance.
    restored = make_model(input_dim=inputs.shape[-1], hidden_dim=32, output_dim=1)
    restored.load_weights(str(weights_path))
    mx.eval(restored.parameters())
    example_pred = restored(inputs[:1])
    print(f"restored prediction sample: {example_pred[0, 0, 0].item():.6f}")


if __name__ == "__main__":
    main()
