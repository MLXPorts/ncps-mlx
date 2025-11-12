"""Shared utilities for MLX examples."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def make_sine_dataset(
    seq_len: int = 64,
    batch_size: int = 1,
    double_frequency: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Generate sine/cosine inputs with optional double-frequency targets."""

    t = mx.linspace(0.0, 2.0 * math.pi, seq_len)
    sines = mx.sin(t)
    cosines = mx.cos(t)
    inputs = mx.stack([sines, cosines], axis=1)  # [seq_len, 2]
    inputs = mx.expand_dims(inputs, axis=0)  # [1, seq_len, 2]
    inputs = mx.broadcast_to(inputs, (batch_size, seq_len, 2))

    factor = 2.0 if double_frequency else 1.0
    targets = mx.sin(factor * t)
    targets = mx.expand_dims(targets, axis=0)
    targets = mx.expand_dims(targets, axis=-1)  # [1, seq_len, 1]
    targets = mx.broadcast_to(targets, (batch_size, seq_len, 1))
    return inputs.astype(mx.float32), targets.astype(mx.float32)


def train_sequence_model(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    apply_constraints: Optional[Callable[[], None]] = None,
    log_interval: int = 25,
) -> None:
    """Train an MLX recurrent model on a fixed mini-batch."""

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(mdl: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        preds, _ = mdl(x)
        return mx.mean((preds - y) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(1, epochs + 1):
        loss, grads = value_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        if apply_constraints is not None:
            apply_constraints()
        mx.eval(model.parameters(), optimizer.state)
        if epoch % log_interval == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss={loss.item():.6f}")
