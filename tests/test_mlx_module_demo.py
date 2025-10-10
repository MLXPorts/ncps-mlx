"""Pytest coverage for the MLX Module training demonstration."""

import math

import mlx.core as mx

from ncps.mlx.module_training_demo import (
    TrainingState,
    create_training_state,
    train_for_epochs,
)


def make_linear_dataset(num_points: int = 64):
    """Create a simple y = 3x + 1 regression dataset."""
    inputs = mx.linspace(0.0, 1.0, num_points, dtype=mx.float32).reshape(num_points, 1)
    targets = 3.0 * inputs + 1.0
    return [(inputs, targets)]


def test_training_converges_on_linear_data():
    """Training loop should reduce loss across epochs."""
    state: TrainingState = create_training_state(
        input_dim=1,
        hidden_dim=16,
        output_dim=1,
        learning_rate=1e-2,
    )

    dataset = make_linear_dataset()
    losses = list(train_for_epochs(state, dataset, epochs=5))

    assert len(losses) == 5
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
