"""Cheetah trajectory prediction example using MLX CfC modules.

This mirrors the classic LTC experiments but runs entirely on MLX arrays, with
no NumPy dependencies. The script expects the cheetah ``.npy`` traces to live
under ``datasets/data/cheetah`` (run ``datasets/download_datasets.sh`` and copy
the traces into that directory). It trains a next-step predictor and
demonstrates saving/loading weights.
"""

from __future__ import annotations

import ast
import struct
from array import array
from pathlib import Path
from typing import Iterable, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import CfC
from ncps.wirings import FullyConnected

DATA_ROOT = Path("datasets/data/cheetah")


def _load_npy(path: Path) -> mx.array:
    """Lightweight NumPy .npy loader that returns an ``mx.array``."""
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file")
        major, minor = struct.unpack("<BB", handle.read(2))
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version: {major}.{minor}")
        header = handle.read(header_len).decode("latin1").strip()
        header_dict = ast.literal_eval(header)
        dtype_descr = header_dict["descr"]
        if dtype_descr not in {"<f4", "<f8"}:
            raise ValueError(f"Unsupported dtype {dtype_descr} in {path}")
        if header_dict.get("fortran_order", False):
            raise ValueError("Fortran-order arrays are not supported in this loader.")
        shape = header_dict["shape"]
        raw = handle.read()

    typecode = "f" if dtype_descr.endswith("4") else "d"
    values = array(typecode)
    values.frombytes(raw)
    floats = [float(v) for v in values]
    arr = mx.array(floats, dtype=mx.float32)
    return mx.reshape(arr, shape)


def _slice_sequences(
    trace: mx.array,
    seq_len: int,
    stride: int,
) -> Tuple[List[mx.array], List[mx.array]]:
    """Create overlapping input/target windows from a single trace."""
    timesteps = trace.shape[0]
    inputs: List[mx.array] = []
    targets: List[mx.array] = []
    for start in range(0, timesteps - seq_len - 1, stride):
        window = trace[start : start + seq_len]
        target = trace[start + 1 : start + seq_len + 1]
        inputs.append(window)
        targets.append(target)
    return inputs, targets


def _collate(traces: Iterable[Path], seq_len: int, stride: int) -> Tuple[mx.array, mx.array]:
    all_inputs: List[mx.array] = []
    all_targets: List[mx.array] = []
    for path in traces:
        trace = _load_npy(path)
        windows, targets = _slice_sequences(trace, seq_len=seq_len, stride=stride)
        if windows:
            all_inputs.extend(windows)
            all_targets.extend(targets)
    if not all_inputs:
        raise ValueError("No sequences produced from cheetah traces.")
    inputs = mx.stack(all_inputs, axis=0)
    targets = mx.stack(all_targets, axis=0)
    return inputs, targets


def load_cheetah_dataset(
    seq_len: int = 32,
    stride: int = 10,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Load train/validation/test splits from cheetah traces."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Expected cheetah traces under {DATA_ROOT}. Copy the files from the "
            "original LTC repository or mount them here."
        )

    files = sorted(DATA_ROOT.glob("trace_*.npy"))
    if len(files) < 15:
        raise FileNotFoundError(
            f"Found {len(files)} cheetah traces; need at least 15 to match the original splits."
        )

    train_inputs, train_targets = _collate(files[15:25], seq_len, stride)
    test_inputs, test_targets = _collate(files[5:15], seq_len, stride)
    valid_inputs, valid_targets = _collate(files[:5], seq_len, stride)
    return train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets


class CheetahPredictor(nn.Module):
    """Two-layer CfC predictor with an MLX Linear readout."""

    def __init__(self, hidden_dim: int = 128, output_dim: int = 17, input_dim: int = 17) -> None:
        super().__init__()
        wiring = FullyConnected(units=hidden_dim, output_dim=output_dim)
        wiring.build(input_dim)
        self.cfc = CfC(
            input_size=input_dim,
            units=wiring,
            return_sequences=True,
            backbone_units=hidden_dim,
            backbone_layers=1,
            backbone_dropout=0.0,
        )
        _ = self.cfc(mx.zeros((1, 1, input_dim)))
        self.readout = nn.Linear(hidden_dim, output_dim)
        mx.eval(self.cfc.parameters(), self.readout.parameters())

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs, _ = self.cfc(inputs)
        return self.readout(outputs)


def mse_loss(predictions: mx.array, targets: mx.array) -> mx.array:
    return mx.mean((predictions - targets) ** 2)


def train_epoch(
    model: CheetahPredictor,
    optimizer: optim.Adam,
    inputs: mx.array,
    targets: mx.array,
    batch_size: int = 16,
) -> float:
    """Run one optimisation epoch and return the average loss."""
    _ = model(inputs[:1])
    mx.eval(model.parameters())
    value_and_grad = nn.value_and_grad(model, lambda mdl, x, y: mse_loss(mdl(x), y))
    total_loss = 0.0
    batches = 0
    for start in range(0, inputs.shape[0], batch_size):
        end = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]
        loss, grads = value_and_grad(model, batch_inputs, batch_targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.tolist()
        batches += 1
    return total_loss / max(batches, 1)


def evaluate_mae(model: CheetahPredictor, inputs: mx.array, targets: mx.array) -> float:
    predictions = model(inputs)
    error = mx.mean(mx.abs(predictions - targets))
    return error.tolist()


def main():
    mx.random.seed(11)
    dataset = load_cheetah_dataset()
    train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = dataset

    model = CheetahPredictor()
    optimizer = optim.Adam(learning_rate=5e-4)
    mx.eval(model.parameters())

    for epoch in range(1, 6):
        train_loss = train_epoch(model, optimizer, train_inputs, train_targets)
        valid_mae = evaluate_mae(model, valid_inputs, valid_targets)
        print(f"epoch {epoch:02d} | train loss: {train_loss:.4f} | valid MAE: {valid_mae:.4f}")

    weights_path = Path("artifacts") / "cheetah_predictor.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))

    restored = CheetahPredictor()
    restored.load_weights(str(weights_path))
    mx.eval(restored.parameters())
    test_mae = evaluate_mae(restored, test_inputs, test_targets)
    print(f"restored model test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
