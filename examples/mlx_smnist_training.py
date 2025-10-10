"""Sequential MNIST training with an MLX-first LTC implementation.

This script mirrors the original MIT CSAIL LTC training loop:
* time-major Sequential MNIST preprocessing
* Liquid Time-Constant cell with semi-implicit / explicit / Runge-Kutta solvers
* Biophysical parameter constraints enforced after every optimisation step
* Best-validation checkpoint tracking and restoration
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import LTC
from ncps.wirings import FullyConnected

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_ROOT = Path("datasets/data/smnist")


def ensure_dataset(path: Path = DATA_ROOT) -> Path:
    """Download Sequential MNIST archive if missing."""
    path.mkdir(parents=True, exist_ok=True)
    archive = path / "mnist.npz"
    if not archive.exists():
        import urllib.request

        print("[data] downloading MNIST …")
        urllib.request.urlretrieve(MNIST_URL, archive)
    return archive


@dataclass
class SMnistBatch:
    inputs: mx.array  # [batch, seq_len, features]
    targets: mx.array  # [batch]


class SMnistData:
    """Sequential MNIST with train/validation/test splits and time-major storage."""

    def __init__(
        self,
        archive: Path | None = None,
        max_train: int | None = None,
        max_eval: int | None = None,
    ):
        archive = ensure_dataset() if archive is None else archive
        raw = mx.load(archive)

        train_x = raw["x_train"].astype(mx.float32) / 255.0
        train_y = raw["y_train"].astype(mx.int32)
        test_x = raw["x_test"].astype(mx.float32) / 255.0
        test_y = raw["y_test"].astype(mx.int32)

        if max_train is not None:
            train_x = train_x[:max_train]
            train_y = train_y[:max_train]

        train_split = int(train_x.shape[0] * 0.9)
        valid_x = train_x[train_split:]
        valid_y = train_y[train_split:]
        train_x = train_x[:train_split]
        train_y = train_y[:train_split]

        if max_eval is not None:
            valid_x = valid_x[:max_eval]
            valid_y = valid_y[:max_eval]
            test_x = test_x[:max_eval]
            test_y = test_y[:max_eval]

        self.seq_len = train_x.shape[1]
        self.input_dim = train_x.shape[2]

        self.train_x = mx.transpose(train_x, (1, 0, 2))  # [seq, batch, feat]
        self.train_y = train_y
        self.valid_x = mx.transpose(valid_x, (1, 0, 2))
        self.valid_y = valid_y
        self.test_x = mx.transpose(test_x, (1, 0, 2))
        self.test_y = test_y

    def _gather(self, tensor: mx.array, indices: mx.array, axis: int) -> mx.array:
        return mx.take(tensor, indices, axis=axis)

    def iterate_train(self, batch_size: int = 16) -> Iterator[SMnistBatch]:
        total = int(self.train_x.shape[1])
        perm = mx.random.permutation(total)
        perm = perm.astype(mx.int32)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = perm[start:end]
            batch_seq = self._gather(self.train_x, idx, axis=1)  # [seq, batch, feat]
            batch_inputs = mx.transpose(batch_seq, (1, 0, 2))
            batch_targets = self._gather(self.train_y, idx, axis=0)
            yield SMnistBatch(inputs=batch_inputs, targets=batch_targets)

    def full_split(self, split: str) -> SMnistBatch:
        if split == "valid":
            seq, labels = self.valid_x, self.valid_y
        elif split == "test":
            seq, labels = self.test_x, self.test_y
        else:
            seq, labels = self.train_x, self.train_y
        return SMnistBatch(inputs=mx.transpose(seq, (1, 0, 2)), targets=labels)


class SMnistLTCClassifier(nn.Module):
    """Wrapper around an LTC layer with a linear readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 10,
        *,
        solver: str = "semi_implicit",
        ode_unfolds: int = 6,
        input_mapping: str = "affine",
    ) -> None:
        super().__init__()
        if solver not in {"semi_implicit", "explicit", "rk4"}:
            raise ValueError(f"Unsupported solver '{solver}' for MLX LTC port")

        wiring = FullyConnected(units=hidden_dim, output_dim=hidden_dim)
        wiring.build(input_dim)
        self.ltc = LTC(
            input_size=input_dim,
            units=wiring,
            return_sequences=False,
            input_mapping=input_mapping,
            ode_unfolds=ode_unfolds,
        )
        self.readout = nn.Linear(hidden_dim, num_classes)

    def __call__(self, inputs: mx.array) -> mx.array:
        features, _ = self.ltc(inputs)
        return self.readout(features)

    def apply_constraints(self) -> None:
        self.ltc.apply_weight_constraints()


def accuracy_from_logits(logits: mx.array, labels: mx.array) -> float:
    preds = mx.argmax(logits, axis=-1)
    acc = mx.mean((preds == labels).astype(mx.float32))
    return float(acc.item())


def evaluate_model(model: SMnistLTCClassifier, batch: SMnistBatch, batch_size: int = 256) -> Tuple[float, float]:
    model.train(False)
    total_loss = 0.0
    total_acc = 0.0
    total = int(batch.inputs.shape[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inputs = batch.inputs[start:end]
        labels = batch.targets[start:end]
        logits = model(inputs)
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        total_loss += float(loss.item()) * (end - start)
        total_acc += accuracy_from_logits(logits, labels) * (end - start)
    normaliser = max(total, 1)
    return total_loss / normaliser, total_acc / normaliser


def train_ltc(
    *,
    data: SMnistData,
    hidden_dim: int = 64,
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 5e-3,
    log_period: int = 50,
    solver: str = "semi_implicit",
    ode_unfolds: int = 6,
    checkpoint_path: Path | None = None,
) -> Tuple[SMnistLTCClassifier, dict]:
    model = SMnistLTCClassifier(
        input_dim=data.input_dim,
        hidden_dim=hidden_dim,
        solver=solver,
        ode_unfolds=ode_unfolds,
    )
    warm_sample = data.full_split("train").inputs[:1]
    _ = model(warm_sample)
    mx.eval(model.parameters())

    def loss_fn(mdl: SMnistLTCClassifier, x: mx.array, y: mx.array) -> mx.array:
        logits = mdl(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=learning_rate)

    best_valid_acc = -1.0
    best_stats: dict[str, float] = {}
    best_checkpoint = checkpoint_path or Path("artifacts") / "smnist_ltc_weights.npz"
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.0
        running_acc = 0.0
        batches = 0

        for batch in data.iterate_train(batch_size):
            loss, grads = value_and_grad(model, batch.inputs, batch.targets)
            optimizer.update(model, grads)
            model.apply_constraints()
            mx.eval(model.parameters(), optimizer.state)

            running_loss += float(loss.item())
            logits = model(batch.inputs)
            running_acc += accuracy_from_logits(logits, batch.targets)
            batches += 1

        if (epoch % log_period) == 0 or epoch == epochs - 1:
            train_split = data.full_split("train")
            valid_split = data.full_split("valid")
            test_split = data.full_split("test")

            train_loss, train_acc = evaluate_model(model, train_split)
            valid_loss, valid_acc = evaluate_model(model, valid_split)
            test_loss, test_acc = evaluate_model(model, test_split)

            print(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc*100:.2f}% "
                f"| valid loss {valid_loss:.4f} acc {valid_acc*100:.2f}% "
                f"| test loss {test_loss:.4f} acc {test_acc*100:.2f}%"
            )

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_stats = {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc * 100.0),
                    "valid_loss": float(valid_loss),
                    "valid_acc": float(valid_acc * 100.0),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc * 100.0),
                }
                model.save_weights(str(best_checkpoint))

    # Restore best checkpoint
    if best_checkpoint.exists():
        model.load_weights(str(best_checkpoint))
        mx.eval(model.parameters())
    return model, best_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential MNIST with MLX LTC.")
    parser.add_argument("--model", default="ltc", choices=["ltc", "ltc_ex", "ltc_rk"])
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--ode-unfolds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts") / "smnist_ltc_weights.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)

    if args.model == "ltc_rk":
        solver = "runge_kutta"
    elif args.model == "ltc_ex":
        solver = "explicit"
    else:
        solver = "semi_implicit"

    data = SMnistData(max_train=args.max_train)
    _, best_stats = train_ltc(
        data=data,
        hidden_dim=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_period=args.log_period,
        solver=solver,
        ode_unfolds=args.ode_unfolds,
        checkpoint_path=args.checkpoint,
    )
    if best_stats:
        print(
            "best validation epoch {epoch:03.0f}: "
            "train {train_acc:.2f}% | valid {valid_acc:.2f}% | test {test_acc:.2f}%".format(
                **best_stats
            )
        )


if __name__ == "__main__":
    main()
