"""Daily minimum temperature forecasting with MLX LTC, matching the original TF example."""

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps.neurons.ltc4_cell import LTC4Cell

LOOK_BACK = 3
EPOCHS = 100
BATCH_SIZE = 1
LR = 1e-3
UNITS = 10


def load_series() -> mx.array:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    dataframe = pd.read_csv(url, usecols=[1], engine="python")
    return mx.array(dataframe.values.astype("float32"))


def create_dataset(dataset: mx.array, look_back: int) -> Tuple[mx.array, mx.array]:
    data_x, data_y = [], []
    dataset_list = mx.reshape(dataset, (-1,)).tolist()
    for i in range(len(dataset_list) - look_back - 1):
        data_x.append(dataset_list[i : i + look_back])
        data_y.append(dataset_list[i + look_back])
    return mx.array(data_x), mx.array(data_y)


class TemperatureLTCModel(nn.Module):
    def __init__(self, units: int = UNITS) -> None:
        super().__init__()
        self.cell = LTC4Cell(units=units, solver="semi_implicit", ode_unfolds=6)
        self.readout = nn.Linear(units, 1)
        self._init_readout()

    def _init_readout(self) -> None:
        fan_in = self.readout.weight.shape[1]
        fan_out = self.readout.weight.shape[0]
        limit = mx.sqrt(mx.array(6.0 / (fan_in + fan_out)))
        # Avoid mixing Python floats with MLX by scaling a 0..1 draw entirely in MLX
        u = mx.random.uniform(shape=self.readout.weight.shape)
        self.readout.weight = (u * (2 * limit)) - limit
        self.readout.bias = mx.zeros(self.readout.bias.shape, dtype=mx.float32)

    def __call__(self, inputs: mx.array) -> mx.array:
        state = None
        for t in range(inputs.shape[1]):
            step = inputs[:, t, :]
            state, _ = self.cell(step, state)
        return self.readout(state).squeeze(-1)


def prepare_data():
    raw = load_series()
    raw_np = mx.reshape(raw, (-1,)).tolist()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform([[x] for x in raw_np])

    train_size = int(len(scaled) * 0.67)
    train_raw = mx.array(scaled[:train_size])
    test_raw = mx.array(scaled[train_size:])

    train_x, train_y = create_dataset(train_raw, LOOK_BACK)
    test_x, test_y = create_dataset(test_raw, LOOK_BACK)

    train_x = mx.reshape(train_x, (train_x.shape[0], LOOK_BACK, 1)).astype(mx.float32)
    test_x = mx.reshape(test_x, (test_x.shape[0], LOOK_BACK, 1)).astype(mx.float32)

    return (
        scaler,
        train_x,
        train_y,
        test_x,
        test_y,
    )


def train_model(model: TemperatureLTCModel, train_inputs: mx.array, train_targets: mx.array, epochs: int) -> None:
    warm_input = train_inputs[:1]
    _ = model(warm_input)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=LR)
    optimizer.init(model.trainable_parameters())

    def loss_fn(mdl: TemperatureLTCModel, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean(mx.power(preds - yb, 2.0))

    value_and_grad = nn.value_and_grad(model, loss_fn)

    num_samples = train_inputs.shape[0]
    for epoch in range(1, epochs + 1):
        permutation = mx.random.permutation(num_samples)
        epoch_losses = []
        for idx in range(num_samples):
            i = int(permutation[idx].item())
            xb = train_inputs[i : i + 1]
            yb = train_targets[i : i + 1]
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_losses.append(loss.item())
        mean_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        print(f"epoch {epoch:03d} train_loss={mean_loss:.6f}")


def evaluate(model: TemperatureLTCModel, scaler: MinMaxScaler, inputs: mx.array, targets: mx.array):
    preds = []
    for idx in range(inputs.shape[0]):
        preds.append(model(inputs[idx : idx + 1]).item())
    
    # Convert to list for sklearn compatibility
    preds_list = [[p] for p in preds]
    targets_list = [[t] for t in targets.tolist()]

    preds_denorm = scaler.inverse_transform(preds_list)
    targets_denorm = scaler.inverse_transform(targets_list)

    rmse = mean_squared_error(targets_denorm, preds_denorm) ** 0.5
    return rmse, mx.array([p[0] for p in preds_denorm]), mx.array([t[0] for t in targets_denorm])


def run_experiment(plot: bool = False, epochs: int = EPOCHS):
    mx.random.seed(123)
    scaler, train_inputs, train_targets, test_inputs, test_targets = prepare_data()
    model = TemperatureLTCModel()
    mx.eval(model.parameters())
    train_model(model, train_inputs, train_targets, epochs=epochs)

    train_rmse, train_preds, train_truth = evaluate(model, scaler, train_inputs, train_targets)
    test_rmse, test_preds, test_truth = evaluate(model, scaler, test_inputs, test_targets)

    if plot:
        full_series_raw = load_series()
        full_series_list = [[x] for x in mx.reshape(full_series_raw, (-1,)).tolist()]
        full_series = scaler.inverse_transform(full_series_list)
        full_series = [x[0] for x in full_series]
        plt.figure(figsize=(12, 6))
        plt.plot(full_series, label="Original Data")
        plt.plot(range(LOOK_BACK, LOOK_BACK + len(train_preds.tolist())), train_preds.tolist(), label="Train Predictions")
        offset = LOOK_BACK + len(train_preds.tolist()) + LOOK_BACK
        plt.plot(range(offset, offset + len(test_preds.tolist())), test_preds.tolist(), label="Test Predictions")
        plt.xlabel("Days")
        plt.ylabel("Temperature")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"train_rmse": float(train_rmse), "test_rmse": float(test_rmse)}


if __name__ == "__main__":
    print(run_experiment(plot=True))
