"""Train a CfC model on the tile-maze simulated LIDAR sequences (MLX).

Generates expert trajectories in the headless maze environment and trains a
small CfC to imitate steering. Saves weights/config/hx under artifacts/maze_cfc.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import wirings
from ncps.neurons import CfC

try:  # Script vs package import compatibility
    from .wiring_presets import make_sensory_motor_wiring
except ImportError:
    from examples.wiring_presets import make_sensory_motor_wiring
from examples.maze_env import SimConfig, make_batch


ARTIFACTS = "artifacts/maze_cfc"
PROFILE = {
    "name": "cfc_maze",
    "description": "CfC configuration for tile-maze LIDAR imitation",
    "extras": {
        "units": 64,
        "proj_size": 1,
        "return_sequences": True,
        "batch_first": True,
        "mode": "default",
        "activation": "lecun_tanh",
        "backbone_units": 128,
        "backbone_layers": 1,
        "backbone_dropout": 0.0
    }
}


def train_maze(
    episodes: int = 64,
    seq_len: int = 64,
    bins: int = 181,
    lr: float = 1e-3,
    epochs: int = 10,
) -> Dict[str, mx.array]:
    cfg = SimConfig(seq_len=seq_len, bins=bins)
    X, Y = make_batch(cfg, episodes)
    # Pack features [LIDAR + 2]
    X = mx.reshape(X, (X.shape[0], X.shape[1], -1))

    # Build model from profile dict
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(PROFILE, fh, indent=2)

    input_dim = int(X.shape[-1])
    extras = PROFILE["extras"]
    # Use NCPS wiring with a single motor neuron (output_dim=1)
    wiring = make_sensory_motor_wiring(input_dim=input_dim, units=int(extras["units"]), output_dim=1)
    model = CfC(
        input_size=input_dim,
        units=wiring,                 # wiring defines the motor neuron
        proj_size=None,
        return_sequences=bool(extras["return_sequences"]),
        batch_first=bool(extras["batch_first"]),
        mode=str(extras["mode"]),
        activation=str(extras["activation"]),
    )
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(mdl: CfC, xb: mx.array, yb: mx.array) -> mx.array:
        preds, _ = mdl(xb)
        return mx.mean((preds - yb) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def batch_iter(bs: int):
        n = int(X.shape[0])
        mx.random.seed(42)
        idx = mx.random.permutation(n)
        for s in range(0, n, bs):
            sl = idx[s:s + bs]
            yield X[sl], Y[sl]

    metrics: Dict[str, mx.array] = {}
    model.train(True)
    for ep in range(1, epochs + 1):
        sum_loss = mx.array(0.0, dtype=mx.float32)
        n = 0
        for xb, yb in batch_iter(bs=8):
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            sum_loss = sum_loss + loss
            n += 1
        metrics[f"epoch_{ep:03d}"] = sum_loss / mx.array(max(1, n), dtype=mx.float32)
        print("epoch", ep, "loss=", metrics[f"epoch_{ep:03d}"])

    # Save weights and a zero-state hx snapshot
    model.save_weights(os.path.join(ARTIFACTS, "weights.npz"))
    hx = mx.zeros((1, int(extras["units"])), dtype=mx.float32)
    mx.savez(os.path.join(ARTIFACTS, "hx.npz"), hx=hx)
    return metrics


def main():  # pragma: no cover
    train_maze()


if __name__ == "__main__":  # pragma: no cover
    main()
