"""Shared utilities for MLX LTC regression examples."""

from typing import Callable, Generator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import LTC, wirings


ArrayPair = Tuple[mx.array, mx.array]


def series_to_windows(
    series: mx.array,
    look_back: int,
    horizon: int = 1,
) -> ArrayPair:
    """Convert a raw 1D/2D series into supervised windows.

    Args:
        series: Array with shape [time] or [time, features].
        look_back: Number of historical steps per sample.
        horizon: Prediction horizon (defaults to 1 for next-step prediction).

    Returns:
        Tuple of (inputs, targets) where inputs has shape
        [num_samples, look_back, features] and targets has
        shape [num_samples, horizon].
    """

    data = mx.array(series, dtype=mx.float32)
    if data.ndim == 1:
        data = mx.expand_dims(data, axis=1)

    num_steps, num_features = data.shape
    if num_steps <= look_back + horizon:
        raise ValueError("Series is too short for the requested window sizes")

    inputs, targets = [], []
    for idx in range(num_steps - look_back - horizon + 1):
        window = data[idx : idx + look_back]
        future = data[idx + look_back : idx + look_back + horizon]
        inputs.append(window)
        if future.shape[-1] == 1:
            targets.append(mx.squeeze(future, axis=-1))
        else:
            targets.append(future)

    inputs_arr = mx.stack(inputs, axis=0)
    targets_arr = mx.stack(targets, axis=0)
    if targets_arr.ndim == 1:
        targets_arr = mx.expand_dims(targets_arr, axis=1)
    return inputs_arr, targets_arr.astype(mx.float32)


def train_val_split(
    inputs: mx.array,
    targets: mx.array,
    ratio: float = 0.8,
) -> Tuple[ArrayPair, ArrayPair]:
    """Split windowed data into train/validation sets without shuffling."""

    split_idx = int(inputs.shape[0] * ratio)
    train = (inputs[:split_idx], targets[:split_idx])
    val = (inputs[split_idx:], targets[split_idx:])
    return train, val


def batch_iterator(
    inputs: mx.array,
    targets: mx.array,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Yield mini-batches as MLX arrays."""

    num_samples = inputs.shape[0]
    indices = mx.arange(num_samples)
    if shuffle:
        mx.random.seed(seed)
        indices = mx.random.permutation(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield inputs[batch_idx], targets[batch_idx]


def make_regression_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    apply_constraints: Optional[Callable[[], None]] = None,
) -> Callable[[mx.array, mx.array], float]:
    """Create a single training step closure for sequence regression."""

    def loss_fn(mdl: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean(mx.power(preds - yb, 2.0))

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def step(xb: mx.array, yb: mx.array) -> float:
        loss, grads = value_and_grad(model, xb, yb)
        optimizer.update(model, grads)
        if apply_constraints is not None:
            apply_constraints()
        mx.eval(model.parameters(), optimizer.state)
        return loss.item()

    return step


def train_sequence_regressor(
    model: nn.Module,
    train_data: ArrayPair,
    val_data: Optional[ArrayPair] = None,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    log_interval: int = 25,
    apply_constraints: Optional[Callable[[], None]] = None,
) -> None:
    """Train an MLX regression model on windowed time series data."""

    optimizer = optim.Adam(learning_rate=learning_rate)
    train_step = make_regression_loop(model, optimizer, apply_constraints)

    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data if val_data is not None else (None, None)

    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in batch_iterator(
            train_inputs, train_targets, batch_size, shuffle=True, seed=epoch
        ):
            xb = xb.astype(mx.float32)
            yb = yb.astype(mx.float32)
            losses.append(train_step(xb, yb))

        if epoch % log_interval == 0 or epoch == 1:
            train_loss = (sum(losses) / len(losses)) if losses else 0.0
            log_line = f"epoch {epoch:03d} train_loss={train_loss:.6f}"

            if val_inputs is not None and val_inputs.shape[0] > 0:
                preds = model(val_inputs.astype(mx.float32))
                val_loss = mx.mean(mx.power(preds - val_targets, 2.0))
                log_line += f" val_loss={val_loss.item():.6f}"
            print(log_line)


class LTCSequenceRegressor(nn.Module):
    """Convenience wrapper combining an LTC core with a linear readout."""

    def __init__(
        self,
        input_size: int,
        hidden_units: int | wirings.Wiring = 32,
        output_dim: int = 1,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()

        wiring = (
            hidden_units
            if isinstance(hidden_units, wirings.Wiring)
            else wirings.FullyConnected(hidden_units)
        )

        self.return_sequences = return_sequences

        self.core = LTC(
            input_size=input_size,
            units=wiring,
            return_sequences=return_sequences,
        )
        self.readout = nn.Linear(self.core.output_size, output_dim)

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs, _ = self.core(inputs)
        if self.return_sequences:
            batch, steps, features = outputs.shape
            flat = outputs.reshape(batch * steps, features)
            preds = self.readout(flat)
            return preds.reshape(batch, steps, -1)
        return self.readout(outputs)

    def apply_constraints(self) -> None:
        self.core.apply_weight_constraints()
