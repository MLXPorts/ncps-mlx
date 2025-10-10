"""CfC training demo on the ICRA 2020 lidar collision-avoidance dataset."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import IcraCfCCell
from datasets.icra2020_lidar_collision_avoidance import load_data


class LidarModel(nn.Module):
    """Exact ICRA lidar imitation wrapper using the specialised CfC cell.

    Expects inputs shaped [batch, time, lidar_bins + 2]. The last two features
    are the vehicle state scalars; the leading features are the lidar bins.
    """

    def __init__(self, total_input_dim: int, kind: str = "cfc", *, units: int = 64) -> None:
        super().__init__()
        if total_input_dim < 3:
            raise ValueError("total_input_dim must include lidar bins + 2 state dims")
        self.lidar_bins = total_input_dim - 2
        self.state_dim = 2
        if kind == "cfc":
            self.cell = IcraCfCCell(lidar_bins=self.lidar_bins, state_dim=self.state_dim, profile="cfc_icra")
        elif kind == "ctrnn":
            self.cell = IcraCTRNNCell(lidar_bins=self.lidar_bins, state_dim=self.state_dim, units=units, profile="ctrnn_tf")
        else:
            raise ValueError("kind must be 'cfc' or 'ctrnn'")

    def __call__(self, inputs: mx.array) -> mx.array:
        # Split combined inputs into lidar and state streams
        lidar_seq = inputs[..., : self.lidar_bins]
        state_seq = inputs[..., self.lidar_bins : self.lidar_bins + self.state_dim]
        return self.cell(state_seq, lidar_seq)

    def apply_constraints(self) -> None:
        self.cell.apply_constraints()


class LidarCfC(LidarModel):
    """Backwards-compatible CfC model wrapper (original behavior)."""

    def __init__(self, total_input_dim: int) -> None:
        super().__init__(total_input_dim=total_input_dim, kind="cfc")


def prepare_dataset(seq_len: int = 32) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    (train_x, train_y), (test_x, test_y) = load_data(seq_len=seq_len)
    # Already float32 MLX arrays
    return train_x, train_y, test_x, test_y


def batch_iterator(inputs: mx.array, targets: mx.array, batch_size: int, seed: int) -> Tuple[mx.array, mx.array]:
    n = int(inputs.shape[0])
    mx.random.seed(seed)
    idx = mx.random.permutation(n)
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size]
        yield inputs[sl], targets[sl]


def train_model(
    model: LidarModel,
    train_inputs: mx.array,
    train_targets: mx.array,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    log_interval: int = 10,
    *,
    val_inputs: mx.array | None = None,
    val_targets: mx.array | None = None,
    log_csv_path: str | None = None,
    best_path: str | None = None,
) -> Dict[str, float]:
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(mdl: LidarModel, xb: mx.array, yb: mx.array) -> mx.array:
        preds = mdl(xb)
        return mx.mean((preds - yb) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    # Optimizer and model are closed over; compile only over array args
    def _loss_and_grads(xb: mx.array, yb: mx.array):
        return value_and_grad(model, xb, yb)

    try:
        compiled_loss_and_grads = mx.compile(_loss_and_grads)
    except Exception:
        compiled_loss_and_grads = _loss_and_grads

    metrics: Dict[str, mx.array] = {}
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_mae,val_loss,val_mae\n")
    best_mae = mx.array(float("inf"), dtype=mx.float32)
    model.train(True)
    for epoch in range(1, epochs + 1):
        sum_loss = mx.array(0.0, dtype=mx.float32)
        sum_mae = mx.array(0.0, dtype=mx.float32)
        n = 0
        for xb, yb in batch_iterator(train_inputs, train_targets, batch_size, seed=epoch):
            try:
                loss, grads = compiled_loss_and_grads(xb, yb)
            except Exception:
                loss, grads = _loss_and_grads(xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            model.apply_constraints()
            sum_loss = sum_loss + loss
            preds = model(xb).astype(mx.float32)
            sum_mae = sum_mae + mx.mean(mx.abs(preds - yb))
            n += 1
        if epoch % log_interval == 0 or epoch == 1:
            denom = mx.array(max(1, n), dtype=mx.float32)
            metrics[f"epoch_{epoch:03d}"] = sum_loss / denom
            print("epoch", f"{epoch:03d}", "train_loss=", metrics[f"epoch_{epoch:03d}"] , "train_mae=", (sum_mae/denom))
        # Validation + best checkpoint per epoch
        val_loss = None
        val_mae = None
        if val_inputs is not None and val_targets is not None:
            evalm = evaluate(model, val_inputs, val_targets)
            val_loss = evalm["rmse"] * evalm["rmse"]
            val_mae = evalm["mae"]
            if best_path is not None and val_mae < best_mae:
                best_mae = val_mae
                model.cell.save_weights(best_path)
        if log_csv_path:
            with open(log_csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{float(metrics[f'epoch_{epoch:03d}'].tolist())},{float((sum_mae/denom).tolist())},{float((val_loss.tolist() if val_loss is not None else 0.0))},{float((val_mae.tolist() if val_mae is not None else 0.0))}\n"
                )
    return metrics


def evaluate(model: LidarModel, inputs: mx.array, targets: mx.array) -> Dict[str, mx.array]:
    model.eval()
    preds = model(inputs).astype(mx.float32)
    mse = mx.mean((preds - targets) ** 2)
    rmse = mx.sqrt(mse)
    mae = mx.mean(mx.abs(preds - targets))
    return {"rmse": rmse, "mae": mae}


def run_experiment(
    epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 32,
    dataset: Tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
    save_dir: str = "artifacts/icra_cfc",
) -> Dict[str, float]:
    if dataset is None:
        train_x, train_y, test_x, test_y = prepare_dataset(seq_len)
    else:
        train_x, train_y, test_x, test_y = dataset

    # Ensure MLX arrays
    train_x = mx.array(train_x)
    train_y = mx.array(train_y)
    test_x = mx.array(test_x)
    test_y = mx.array(test_y)

    train_x = mx.reshape(train_x, (train_x.shape[0], train_x.shape[1], -1))
    train_y = mx.reshape(train_y, (train_y.shape[0], train_y.shape[1], -1))
    test_x = mx.reshape(test_x, (test_x.shape[0], test_x.shape[1], -1))
    test_y = mx.reshape(test_y, (test_y.shape[0], test_y.shape[1], -1))

    # Simple reproducible split for validation (10% tail of train)
    n = int(train_x.shape[0])
    val_n = max(1, n // 10)
    val_x = train_x[-val_n:]
    val_y = train_y[-val_n:]
    tr_x = train_x[:-val_n]
    tr_y = train_y[:-val_n]

    model = LidarCfC(total_input_dim=train_x.shape[-1])

    # Top-level seed for reproducibility
    mx.random.seed(42)
    # Track best checkpoint by validation MAE
    best_path = os.path.join(save_dir, "best.npz")
    os.makedirs(save_dir, exist_ok=True)
    log_csv = os.path.join(save_dir, "training_log.csv")
    train_metrics = train_model(
        model,
        tr_x,
        tr_y,
        epochs=epochs,
        batch_size=batch_size,
        val_inputs=val_x,
        val_targets=val_y,
        log_csv_path=log_csv,
        best_path=best_path,
    )

    # Restore best
    if os.path.isfile(best_path):
        model.cell.load_weights(best_path)

    eval_metrics = evaluate(model, test_x, test_y)
    print("Test:", eval_metrics)

    # Always save weights, hidden state snapshot, and config JSON
    weights_path = os.path.join(save_dir, "weights.npz")
    model.cell.save_weights(weights_path)

    # Hidden state snapshot from first test sequence (save without helper module)
    sample = test_x[:1]  # [1,T,D]
    lidar_bins = model.lidar_bins
    lidar_seq = sample[..., :lidar_bins]
    state_seq = sample[..., lidar_bins:lidar_bins + model.state_dim]
    _, hx_final = model.cell(state_seq, lidar_seq, return_state=True)
    mx.savez(os.path.join(save_dir, "hx.npz"), hx=hx_final)

    # Config JSON
    cfg = model.cell.to_config()
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    return {"test_rmse": eval_metrics["rmse"], "test_mae": eval_metrics["mae"], **train_metrics}


def main() -> None:
    # Load run configuration from JSON (no CLI flags required)
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "icra_cfc_run.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    run_experiment(
        epochs=int(cfg.get("epochs", 50)),
        batch_size=int(cfg.get("batch_size", 64)),
        seq_len=int(cfg.get("seq_len", 32)),
        save_dir=str(cfg.get("save_dir", "artifacts/icra_cfc")),
    )


if __name__ == "__main__":
    main()
