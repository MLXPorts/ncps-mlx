"""CfC training demo on the ICRA 2020 lidar collision-avoidance dataset."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import CfC
from ncps.datasets.icra2020_lidar_collision_avoidance import load_data


class LidarCfC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_units: int = 64) -> None:
        super().__init__()
        self.rnn = CfC(
            input_size=input_dim,
            units=hidden_units,
            proj_size=output_dim,
            return_sequences=True,
            batch_first=True,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs, _ = self.rnn(inputs)
        return outputs

    def apply_constraints(self) -> None:
        if hasattr(self.rnn.rnn_cell, "apply_weight_constraints"):
            self.rnn.rnn_cell.apply_weight_constraints()


def prepare_dataset(seq_len: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (train_x, train_y), (test_x, test_y) = load_data(seq_len=seq_len)
    return train_x.astype(np.float32), train_y.astype(np.float32), test_x.astype(np.float32), test_y.astype(np.float32)


def batch_iterator(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seed: int) -> Tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(inputs.shape[0])
    for start in range(0, inputs.shape[0], batch_size):
        idx = indices[start:start + batch_size]
        yield mx.array(inputs[idx]), mx.array(targets[idx])


def train_model(
    model: LidarCfC,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    log_interval: int = 10,
) -> Dict[str, float]:
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(mdl: LidarCfC, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean((preds - yb) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    metrics: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in batch_iterator(train_inputs, train_targets, batch_size, seed=epoch):
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            model.apply_constraints()
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss.item()))
        if epoch % log_interval == 0 or epoch == 1:
            metrics[f"epoch_{epoch:03d}"] = float(np.mean(losses))
            print(f"epoch {epoch:03d} train_loss={metrics[f'epoch_{epoch:03d}']:.6f}")
    return metrics


def evaluate(model: LidarCfC, inputs: np.ndarray, targets: np.ndarray) -> float:
    preds = model(mx.array(inputs)).astype(mx.float32)
    mse = mx.mean((preds - mx.array(targets)) ** 2)
    return float(mx.sqrt(mse).item())


def run_experiment(
    epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 32,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Dict[str, float]:
    if dataset is None:
        train_x, train_y, test_x, test_y = prepare_dataset(seq_len)
    else:
        train_x, train_y, test_x, test_y = dataset

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1)
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], -1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], -1)
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], -1)

    model = LidarCfC(input_dim=train_x.shape[-1], output_dim=train_y.shape[-1])
    train_metrics = train_model(model, train_x, train_y, epochs=epochs, batch_size=batch_size)
    test_rmse = evaluate(model, test_x, test_y)
    print(f"Test RMSE: {test_rmse:.4f}")
    return {"test_rmse": test_rmse, **train_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CfC on the ICRA 2020 lidar dataset using MLX.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    args = parser.parse_args()
    run_experiment(epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
