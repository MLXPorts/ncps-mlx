"""MLX port of the USD/TRY LTC currency predictor (ltc_tensor.py)."""

import math
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps.neurons.ltcse_cell import LTCSECell

SEQ_LEN = 32
BATCH_SIZE = 16
LR = 1e-2  # LTC uses higher learning rate in the original script
UNITS = 64
EPOCHS = 500
LOG_PERIOD = 50
SEED = 23489


def convert_to_floats(feature_col: list[str], memory: list[float]) -> Tuple[list[float], list[float]]:
    for i in range(len(feature_col)):
        if feature_col[i] in {"?", "\n"}:
            feature_col[i] = memory[i]
        else:
            value = float(feature_col[i])
            feature_col[i] = value
            memory[i] = value
    return feature_col, memory


def load_crappy_formatted_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    memory = None
    with path.open("r") as handle:
        for lineno, raw in enumerate(handle):
            if lineno == 0:
                continue
            cleaned = raw.strip()
            if not cleaned:
                continue
            cleaned = cleaned.replace('""', '"').strip('"')
            parts = cleaned.split('","')
            if len(parts) < 5:
                continue
            date_price = parts[0].split(',"')
            if len(date_price) != 2:
                continue
            price = date_price[1]
            rest = parts[1:]
            row = [price] + rest
            floats = []
            if memory is None:
                memory = [float(i) for i in range(len(row))]
            for idx, value in enumerate(row):
                value = value.replace('%', '').strip()
                if value in {"", "?"}:
                    floats.append(memory[idx])
                else:
                    val = float(value)
                    floats.append(val)
                    memory[idx] = val
            rows.append(floats)

    if not rows:
        raise RuntimeError(f"No usable data in {path}")

    all_x = np.array(rows, dtype=np.float32)
    mean = np.mean(all_x, axis=0, keepdims=True)
    std = np.std(all_x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    normalized = (all_x - mean) / std

    targets = normalized[:, 0:1]
    features = normalized[:, 1:]
    return features, targets, mean, std


def cut_in_sequences(x: np.ndarray, y: np.ndarray, seq_len: int, inc: int = 1):
    sequences_x = []
    sequences_y = []
    for start in range(0, x.shape[0] - seq_len, inc):
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class PredictionData:
    def __init__(self, data_path: Path, seq_len: int = SEQ_LEN):
        x, y, mean, std = load_crappy_formatted_csv(data_path)
        self.mean = mean
        self.std = std
        self.train_x, self.train_y = cut_in_sequences(x, y, seq_len, inc=seq_len)

        total_seqs = self.train_x.shape[1]
        rng = np.random.RandomState(SEED)
        permutation = rng.permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]

    def iterate_train(self, batch_size: int = BATCH_SIZE) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = max(1, total_seqs // batch_size)
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield batch_x, batch_y


def clone_tree(tree):
    if isinstance(tree, dict):
        return {k: clone_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [clone_tree(v) for v in tree]
    if isinstance(tree, mx.array):
        return mx.array(tree)
    return tree


class CurrencyLTCPredictor(nn.Module):
    def __init__(self, feature_dim: int, units: int = UNITS) -> None:
        super().__init__()
        self.cell = LTCSECell(units=units)
        self.readout = nn.Linear(units, 1)
        self._init_readout()

    def _init_readout(self) -> None:
        stddev = 0.05
        weight = stddev * mx.random.normal(shape=self.readout.weight.shape)
        weight = mx.clip(weight, -2 * stddev, 2 * stddev)
        self.readout.weight = weight
        self.readout.bias = mx.zeros(self.readout.bias.shape, dtype=mx.float32)

    def __call__(self, inputs: mx.array) -> mx.array:
        time_steps, batch_size, _ = inputs.shape
        state = None
        outputs = []
        for t in range(time_steps):
            step = inputs[t]
            state, _ = self.cell(step, state)
            outputs.append(self.readout(state).reshape(1, batch_size, 1))
        return mx.concatenate(outputs, axis=0)

    def apply_constraints(self) -> None:
        self.cell.apply_constraints()
        self.readout.weight = mx.clip(self.readout.weight, -0.5, 0.5)
        self.readout.bias = mx.clip(self.readout.bias, -0.5, 0.5)


def to_time_major(batch_x: np.ndarray) -> mx.array:
    return mx.array(batch_x.astype(np.float32))


def inverse_scale(values: mx.array, mean: np.ndarray, std: np.ndarray, index: int = 0) -> np.ndarray:
    vals = np.array(values.tolist(), dtype=np.float32)
    return vals * std[0, index] + mean[0, index]


def run_experiment(epochs: int = EPOCHS, plot: bool = True) -> dict[str, float]:
    data_path = Path(__file__).resolve().parent / "data" / "USDTRY_Data.csv"
    dataset = PredictionData(data_path)

    model = CurrencyLTCPredictor(feature_dim=dataset.train_x.shape[2])

    # Warm up parameters
    warm_input = to_time_major(dataset.train_x[:, :1, :])
    _ = model(warm_input)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=LR)
    optimizer.init(model.trainable_parameters())

    def loss_fn(mdl: CurrencyLTCPredictor, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean(mx.power(preds - yb, 2.0))

    value_and_grad = nn.value_and_grad(model, loss_fn)

    best_val_loss = math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        losses = []
        for batch_x, batch_y in dataset.iterate_train(BATCH_SIZE):
            xb = to_time_major(batch_x)
            yb = mx.array(batch_y.astype(np.float32))
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            model.apply_constraints()
            losses.append(float(loss.item()))

        if epoch % LOG_PERIOD == 0 or epoch == 1:
            val_inputs = to_time_major(dataset.valid_x)
            val_targets = mx.array(dataset.valid_y.astype(np.float32))
            val_loss = loss_fn(model, val_inputs, val_targets)
            val_loss_value = float(val_loss.item())
            mean_train_loss = float(mx.mean(mx.array(losses)).item())
            print(f"epoch {epoch:03d} train_loss={mean_train_loss:.6f} val_loss={val_loss_value:.6f}")
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_state = clone_tree(model.parameters())

    if best_state is not None:
        model.update(best_state)
        mx.eval(model.parameters())

    def evaluate(split_x: np.ndarray, split_y: np.ndarray) -> Tuple[float, float]:
        preds = model(to_time_major(split_x))
        preds_np = inverse_scale(preds, dataset.mean, dataset.std, index=0)
        targets_np = inverse_scale(mx.array(split_y.astype(np.float32)), dataset.mean, dataset.std, index=0)
        preds_flat = preds_np.reshape(-1)
        targets_flat = targets_np.reshape(-1)
        # Use MLX for compute-heavy ops where possible
        diff = preds_flat - targets_flat
        rmse = math.sqrt(float(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        return rmse, mae, preds_np, targets_np

    train_rmse, train_mae, train_preds, train_truth = evaluate(dataset.train_x, dataset.train_y)
    val_rmse, val_mae, val_preds, val_truth = evaluate(dataset.valid_x, dataset.valid_y)
    test_rmse, test_mae, test_preds, test_truth = evaluate(dataset.test_x, dataset.test_y)

    if plot:
        plt.figure(figsize=(12, 4))
        sample_idx = 0
        plt.plot(test_truth[:, sample_idx, 0], label="Ground Truth")
        plt.plot(test_preds[:, sample_idx, 0], label="Prediction")
        plt.title("USD/TRY Sequence Forecast")
        plt.xlabel("Time step")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Train RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

    return {
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
    }


if __name__ == "__main__":
    run_experiment()
