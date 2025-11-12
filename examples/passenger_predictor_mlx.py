"""MLX reimplementation of the original airline-passenger LTC experiment.

Matches the historical TensorFlow script:
- MinMax scaling in [0, 1]
- look_back = 3 windows
- Training split 67% / 33%
- Batch size 1, epochs 100, validation split 0.1
- Single LTC4 cell (32 units) feeding a dense readout
"""

from dataclasses import dataclass
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
UNITS = 32
VALIDATION_SPLIT = 0.1


def load_series() -> mx.array:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    dataframe = pd.read_csv(url, usecols=[1], engine="python")
    return mx.array(dataframe.values.astype("float32"))


def create_dataset(sequence: mx.array, look_back: int) -> Tuple[mx.array, mx.array]:
    data_x, data_y = [], []
    sequence_list = mx.reshape(sequence, (-1,)).tolist()
    for i in range(len(sequence_list) - look_back - 1):
        window = sequence_list[i : i + look_back]
        data_x.append(window)
        data_y.append(sequence_list[i + look_back])
    return mx.array(data_x), mx.array(data_y)


class PassengerLTCModel(nn.Module):
    def __init__(self, look_back: int, units: int = UNITS) -> None:
        super().__init__()
        self.cell = LTC4Cell(units=units, solver="semi_implicit", ode_unfolds=6)
        self.readout = nn.Linear(units, 1)
        self._init_readout()
        self.look_back = look_back

    def _init_readout(self) -> None:
        fan_in = self.readout.weight.shape[1]
        fan_out = self.readout.weight.shape[0]
        limit = mx.sqrt(mx.array(6.0 / (fan_in + fan_out)))
        # Draw in MLX and scale/shift to [-limit, limit] without Python floats
        u = mx.random.uniform(shape=self.readout.weight.shape)
        self.readout.weight = (u * (2 * limit)) - limit
        self.readout.bias = mx.zeros(self.readout.bias.shape, dtype=mx.float32)

    def __call__(self, inputs: mx.array) -> mx.array:
        # inputs: [batch, timesteps, features]
        state = None
        for t in range(inputs.shape[1]):
            step = inputs[:, t, :]
            state, _ = self.cell(step, state)
        return self.readout(state).squeeze(-1)


@dataclass
class DataBundle:
    scaler: MinMaxScaler
    train_inputs: mx.array
    train_targets: mx.array
    val_inputs: mx.array
    val_targets: mx.array
    test_inputs: mx.array
    test_targets: mx.array


def prepare_data() -> DataBundle:
    raw = load_series()
    raw_list = [[x] for x in mx.reshape(raw, (-1,)).tolist()]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw_list)

    train_size = int(len(scaled) * 0.67)
    train_raw = mx.array(scaled[:train_size])
    test_raw = mx.array(scaled[train_size:])

    train_x, train_y = create_dataset(train_raw, LOOK_BACK)
    test_x, test_y = create_dataset(test_raw, LOOK_BACK)

    train_x = mx.reshape(train_x, (train_x.shape[0], LOOK_BACK, 1)).astype(mx.float32)
    test_x = mx.reshape(test_x, (test_x.shape[0], LOOK_BACK, 1)).astype(mx.float32)

    val_start = int(train_x.shape[0] * (1 - VALIDATION_SPLIT))
    val_inputs = train_x[val_start:]
    val_targets = train_y[val_start:]
    train_inputs = train_x[:val_start]
    train_targets = train_y[:val_start]

    return DataBundle(
        scaler=scaler,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        test_inputs=test_x,
        test_targets=test_y,
    )


def train_model(model: PassengerLTCModel, data: DataBundle, epochs: int) -> None:
    train_inputs = data.train_inputs
    train_targets = data.train_targets
    warm_input = train_inputs[:1]
    _ = model(warm_input)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=LR)
    optimizer.init(model.trainable_parameters())

    def loss_fn(mdl: PassengerLTCModel, xb: mx.array, yb: mx.array) -> mx.array:
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

        if data.val_inputs.shape[0] > 0:
            val_loss = loss_fn(model, data.val_inputs, data.val_targets)
            mean_train_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
            print(
                f"epoch {epoch:03d} train_loss={mean_train_loss:.6f} "
                f"val_loss={val_loss.item():.6f}"
            )


def evaluate(model: PassengerLTCModel, data: DataBundle, scaler: MinMaxScaler):
    def predict(inputs: mx.array) -> mx.array:
        outputs = []
        for idx in range(inputs.shape[0]):
            preds = model(inputs[idx : idx + 1])
            outputs.append(preds.item())
        return mx.array(outputs)

    train_preds = predict(data.train_inputs)
    test_preds = predict(data.test_inputs)

    train_targets_list = [[t] for t in data.train_targets.tolist()]
    test_targets_list = [[t] for t in data.test_targets.tolist()]
    train_preds_list = [[p] for p in train_preds.tolist()]
    test_preds_list = [[p] for p in test_preds.tolist()]

    train_denorm = scaler.inverse_transform(train_preds_list)
    test_denorm = scaler.inverse_transform(test_preds_list)
    train_target_denorm = scaler.inverse_transform(train_targets_list)
    test_target_denorm = scaler.inverse_transform(test_targets_list)

    train_rmse = mean_squared_error([t[0] for t in train_target_denorm], [t[0] for t in train_denorm]) ** 0.5
    test_rmse = mean_squared_error([t[0] for t in test_target_denorm], [t[0] for t in test_denorm]) ** 0.5

    return {
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_predictions": mx.array([t[0] for t in train_denorm]),
        "train_targets": mx.array([t[0] for t in train_target_denorm]),
        "test_predictions": mx.array([t[0] for t in test_denorm]),
        "test_targets": mx.array([t[0] for t in test_target_denorm]),
    }


def run_experiment(plot: bool = False, epochs: int = EPOCHS):
    mx.random.seed(42)
    data = prepare_data()
    model = PassengerLTCModel(look_back=LOOK_BACK)
    mx.eval(model.parameters())
    train_model(model, data, epochs=epochs)
    metrics = evaluate(model, data, data.scaler)

    if plot:
        series_raw = mx.reshape(load_series(), (-1,)).tolist()
        series = data.scaler.inverse_transform([[x] for x in series_raw]).squeeze(-1)
        train_len = metrics["train_predictions"].shape[0]
        plt.figure(figsize=(12, 6))
        plt.plot(series, label="True Data")
        plt.plot(range(LOOK_BACK, LOOK_BACK + train_len), metrics["train_predictions"], label="Train Predictions")
        offset = LOOK_BACK + train_len + LOOK_BACK
        plt.plot(range(offset, offset + metrics["test_predictions"].shape[0]), metrics["test_predictions"], label="Test Predictions")
        plt.xlabel("Months")
        plt.ylabel("Airline Passengers")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"train_rmse": metrics["train_rmse"], "test_rmse": metrics["test_rmse"]}


if __name__ == "__main__":
    print(run_experiment(plot=True))
