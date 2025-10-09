"""Daily minimum temperature forecasting with MLX LTC, matching the original TF example."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps.ncps_mlx.ltc4_cell import LTC4Cell

LOOK_BACK = 3
EPOCHS = 100
BATCH_SIZE = 1
LR = 1e-3
UNITS = 10


def load_series() -> np.ndarray:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    dataframe = pd.read_csv(url, usecols=[1], engine="python")
    return dataframe.values.astype("float32")


def create_dataset(dataset: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        data_x.append(dataset[i : i + look_back, 0])
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


class TemperatureLTCModel(nn.Module):
    def __init__(self, units: int = UNITS) -> None:
        super().__init__()
        self.cell = LTC4Cell(units=units, solver="semi_implicit", ode_unfolds=6)
        self.readout = nn.Linear(units, 1)
        self._init_readout()

    def _init_readout(self) -> None:
        fan_in = self.readout.weight.shape[1]
        fan_out = self.readout.weight.shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.readout.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=self.readout.weight.shape,
        )
        self.readout.bias = mx.zeros(self.readout.bias.shape, dtype=mx.float32)

    def __call__(self, inputs: mx.array) -> mx.array:
        state = None
        for t in range(inputs.shape[1]):
            step = inputs[:, t, :]
            state, _ = self.cell(step, state)
        return self.readout(state).squeeze(-1)


def prepare_data():
    raw = load_series()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)

    train_size = int(len(scaled) * 0.67)
    train_raw = scaled[:train_size]
    test_raw = scaled[train_size:]

    train_x, train_y = create_dataset(train_raw, LOOK_BACK)
    test_x, test_y = create_dataset(test_raw, LOOK_BACK)

    train_x = train_x.reshape((train_x.shape[0], LOOK_BACK, 1)).astype(np.float32)
    test_x = test_x.reshape((test_x.shape[0], LOOK_BACK, 1)).astype(np.float32)

    return (
        scaler,
        mx.array(train_x),
        mx.array(train_y),
        mx.array(test_x),
        mx.array(test_y),
    )


def train_model(model: TemperatureLTCModel, train_inputs: mx.array, train_targets: mx.array, epochs: int) -> None:
    warm_input = train_inputs[:1]
    _ = model(warm_input)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=LR)
    optimizer.init(model.trainable_parameters())

    def loss_fn(mdl: TemperatureLTCModel, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean((preds - yb) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    num_samples = train_inputs.shape[0]
    for epoch in range(1, epochs + 1):
        permutation = np.random.permutation(num_samples)
        epoch_losses = []
        for idx in permutation:
            i = int(idx)
            xb = train_inputs[i : i + 1]
            yb = train_targets[i : i + 1]
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_losses.append(float(loss.item()))
        print(f"epoch {epoch:03d} train_loss={np.mean(epoch_losses):.6f}")


def evaluate(model: TemperatureLTCModel, scaler: MinMaxScaler, inputs: mx.array, targets: mx.array):
    preds = []
    for idx in range(inputs.shape[0]):
        preds.append(float(model(inputs[idx : idx + 1]).item()))
    preds = np.array(preds, dtype=np.float32)

    preds_denorm = scaler.inverse_transform(preds.reshape(-1, 1))
    targets_denorm = scaler.inverse_transform(np.array(targets.tolist(), dtype=np.float32).reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(targets_denorm[:, 0], preds_denorm[:, 0]))
    return rmse, preds_denorm.squeeze(-1), targets_denorm.squeeze(-1)


def run_experiment(plot: bool = False, epochs: int = EPOCHS):
    mx.random.seed(123)
    scaler, train_inputs, train_targets, test_inputs, test_targets = prepare_data()
    model = TemperatureLTCModel()
    mx.eval(model.parameters())
    train_model(model, train_inputs, train_targets, epochs=epochs)

    train_rmse, train_preds, train_truth = evaluate(model, scaler, train_inputs, train_targets)
    test_rmse, test_preds, test_truth = evaluate(model, scaler, test_inputs, test_targets)

    if plot:
        full_series = scaler.inverse_transform(load_series()).squeeze(-1)
        plt.figure(figsize=(12, 6))
        plt.plot(full_series, label="Original Data")
        plt.plot(range(LOOK_BACK, LOOK_BACK + len(train_preds)), train_preds, label="Train Predictions")
        offset = LOOK_BACK + len(train_preds) + LOOK_BACK
        plt.plot(range(offset, offset + len(test_preds)), test_preds, label="Test Predictions")
        plt.xlabel("Days")
        plt.ylabel("Temperature")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"train_rmse": float(train_rmse), "test_rmse": float(test_rmse)}


if __name__ == "__main__":
    print(run_experiment(plot=True))
