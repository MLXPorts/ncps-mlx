"""Demonstrations of modern MLX ``nn.Module`` usage and training patterns.

These utilities showcase how to structure models with ``mlx.nn.Module`` and
how to run gradient-based optimization using ``mlx.nn.value_and_grad`` together
with ``mlx.optimizers``. The goal is to provide simple, well-tested examples
that map directly onto the current MLX API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class LiquidRegressor(nn.Module):
    """A minimal regression head using MLX ``Module`` composition."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def __call__(self, inputs: mx.array) -> mx.array:
        encoded = self.encoder(inputs)
        return self.head(encoded)


@dataclass(frozen=True)
class TrainingState:
    """Holds the model and optimizer for convenience."""

    model: LiquidRegressor
    optimizer: optim.Adam


def create_training_state(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    learning_rate: float = 1e-3,
) -> TrainingState:
    """Initialise a model and matching Adam optimizer."""
    model = LiquidRegressor(input_dim, hidden_dim, output_dim)
    mx.eval(model.parameters())  # ensure parameters are materialised
    optimizer = optim.Adam(learning_rate=learning_rate)
    return TrainingState(model=model, optimizer=optimizer)


def mse_loss(predictions: mx.array, targets: mx.array) -> mx.array:
    """Mean-squared error helper."""
    return mx.mean((predictions - targets) ** 2)


def training_step(
    state: TrainingState,
    inputs: mx.array,
    targets: mx.array,
    loss_fn: Callable[[mx.array, mx.array], mx.array] = mse_loss,
) -> Tuple[TrainingState, float]:
    """Run a single optimisation step."""

    def forward(batch_inputs: mx.array, batch_targets: mx.array):
        predictions = state.model(batch_inputs)
        return loss_fn(predictions, batch_targets)

    value_and_grad_fn = nn.value_and_grad(state.model, forward)
    loss, grads = value_and_grad_fn(inputs, targets)
    state.optimizer.update(state.model, grads)
    mx.eval(state.model.parameters(), state.optimizer.state)
    return state, float(loss.item())


def train_for_epochs(
    state: TrainingState,
    dataset: Iterable[Tuple[mx.array, mx.array]],
    epochs: int = 5,
) -> Iterable[float]:
    """Train over the provided dataset and yield losses per epoch."""
    for _ in range(epochs):
        running_loss = 0.0
        batches = 0
        for batch_inputs, batch_targets in dataset:
            state, batch_loss = training_step(state, batch_inputs, batch_targets)
            running_loss += batch_loss
            batches += 1
        yield running_loss / max(batches, 1)
