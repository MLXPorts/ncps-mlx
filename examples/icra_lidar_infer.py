"""Inference utilities for the ICRA lidar regression model (MLX).

Usage:
    PYTHONPATH=. python examples/icra_lidar_infer.py --weights path/to.npz \
        --steps 8
"""

from __future__ import annotations

import argparse
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from ncps import IcraCfCCell
from ncps.ncps_mlx.datasets.icra2020_lidar_collision_avoidance import load_data


def build_model(total_input_dim: int) -> IcraCfCCell:
    lidar_bins = total_input_dim - 2
    return IcraCfCCell(lidar_bins=lidar_bins, state_dim=2, profile="cfc_icra")


def predict_sequence(
    model: IcraCfCCell,
    packed_seq: mx.array,  # [T, L+2]
    hx: mx.array | None = None,
) -> Tuple[mx.array, mx.array]:
    L = model.lidar_bins
    state_seq = packed_seq[:, L:]
    lidar_seq = packed_seq[:, :L]
    state_seq = mx.expand_dims(state_seq, axis=0)    # [1,T,2]
    lidar_seq = mx.expand_dims(lidar_seq, axis=0)    # [1,T,L]
    y_seq, hx_next = model(state_seq, lidar_seq, hx=hx, return_state=True)
    return y_seq[0].astype(mx.float32), hx_next


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=False, default=None)
    ap.add_argument("--hx", type=str, required=False, default=None)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    # Load a tiny real batch from the dataset
    (train_x, train_y), (test_x, test_y) = load_data(seq_len=args.steps)
    x = train_x[0]
    x = mx.reshape(x, (x.shape[0], -1))

    model = build_model(total_input_dim=x.shape[-1])
    if args.weights:
        model.load_weights(args.weights)

    hx = None
    if args.hx:
        hx = mx.load(args.hx)["hx"]

    # Run inference over the sequence
    y_hat, _ = predict_sequence(model, x, hx=hx)
    # Only print shape to avoid host conversions
    print("Predictions shape:", y_hat.shape)


if __name__ == "__main__":
    main()
