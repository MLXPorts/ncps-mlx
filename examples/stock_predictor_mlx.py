"""Stock prediction example using the MLX port of LTC4.

This script mirrors experiments_with_ltcs/StockPredictor/LTC_SP.py
from michaelkhany/liquid_time_constant_networks but runs entirely on MLX.
"""

import argparse
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ncps.neurons import LTC4


def load_stock_data(symbol: str, start: str, end: str) -> np.ndarray:
    yf.pdr_override()
    df = pdr.get_data_yahoo(symbol, start=start, end=end)
    return df[["Close"]].values.astype(np.float32)


def create_dataset(dataset: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i : i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class StockModel(nn.Module):
    def __init__(self, input_len: int, hidden_units: int = 50) -> None:
        super().__init__()
        self.rnn = LTC4(input_size=1, units=hidden_units, return_sequences=False)
        self.readout = nn.Linear(hidden_units, 1)
        self.input_len = input_len

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, time, features]
        features = self.rnn(x)
        return self.readout(features)


def train(model: StockModel, train_x: mx.array, train_y: mx.array, epochs: int, batch_size: int) -> None:
    optimizer = optim.Adam(1e-3)
    loss_fn = nn.losses.mse_loss
    value_and_grad = nn.value_and_grad(model, lambda mdl, inputs, targets: loss_fn(mdl(inputs), targets))

    num_samples = train_x.shape[0]

    for epoch in range(epochs):
        indices = mx.random.permutation(num_samples)
        losses = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            loss, grads = value_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            mean_loss = (sum(losses) / len(losses)) if losses else 0.0
            print(f"epoch {epoch + 1:03d} loss={mean_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX LTC4 stock predictor")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2021-09-01")
    parser.add_argument("--look_back", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    raw_data = load_stock_data(args.symbol, args.start, args.end)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw_data)

    train_size = int(len(scaled) * 0.8)
    train, test = scaled[:train_size], scaled[train_size:]

    X_train, y_train = create_dataset(train, args.look_back)
    X_test, y_test = create_dataset(test, args.look_back)

    X_train = X_train.reshape((-1, args.look_back, 1))
    X_test = X_test.reshape((-1, args.look_back, 1))

    train_x = mx.array(X_train)
    train_y = mx.array(y_train).reshape((-1, 1))
    test_x = mx.array(X_test)
    test_y = mx.array(y_test).reshape((-1, 1))

    model = StockModel(input_len=args.look_back, hidden_units=50)
    mx.eval(model.parameters())

    train(model, train_x, train_y, epochs=args.epochs, batch_size=args.batch_size)

    train_pred = model(train_x)
    test_pred = model(test_x)
    mx.eval(train_pred, test_pred)

    train_pred_np = scaler.inverse_transform(train_pred.tolist())
    train_true_np = scaler.inverse_transform(train_y.tolist())
    test_pred_np = scaler.inverse_transform(test_pred.tolist())
    test_true_np = scaler.inverse_transform(test_y.tolist())

    train_rmse = mean_squared_error(train_true_np, train_pred_np, squared=False)
    test_rmse = mean_squared_error(test_true_np, test_pred_np, squared=False)
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
