"""Stacked MLX NCP example on synthetic lidar-like data."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps.neurons import LTC
from ncps.wirings import AutoNCP


def make_lidar_batch(batch: int, time_steps: int, num_bins: int) -> mx.array:
    angles = mx.linspace(-math.pi, math.pi, num_bins)
    base = mx.expand_dims(mx.cos(angles), axis=0)
    base = mx.expand_dims(base, axis=0)
    base = mx.broadcast_to(base, (batch, time_steps, num_bins))
    noise = 0.1 * mx.random.uniform(shape=base.shape)
    return base + noise


class LidarStack(nn.Module):
    def __init__(self, num_bins: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(num_bins, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.rnn = LTC(
            input_size=32,
            units=AutoNCP(units=20, output_size=1, sparsity_level=0.5),
            return_sequences=True,
        )

    def __call__(self, inputs: mx.array):
        batch, seq, features = inputs.shape
        flat = mx.reshape(inputs, (batch * seq, features))
        pre = self.pre(flat)
        pre = mx.reshape(pre, (batch, seq, -1))
        return self.rnn(pre)


def main() -> None:
    batch = 8
    seq = 12
    bins = 64
    inputs = make_lidar_batch(batch, seq, bins)
    targets = mx.sin(mx.linspace(0, 3 * math.pi, seq))
    targets = mx.expand_dims(mx.expand_dims(targets, axis=0), axis=-1)
    targets = mx.broadcast_to(targets, (batch, seq, 1))

    model = LidarStack(num_bins=bins)
    mx.eval(model.parameters())

    def loss_fn(mdl, x, y):
        preds, _ = mdl(x)
        return mx.mean((preds - y) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=3e-3)

    for epoch in range(150):
        loss, grads = value_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        model.rnn.apply_weight_constraints()
        mx.eval(model.parameters(), optimizer.state)
        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch+1:03d} loss={float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
