"""Heuristic collision-avoidance evaluator for the ICRA lidar model (MLX).

This runs the model in streaming mode over a real sequence and checks whether
predicted steering aligns with the safer side when the front sector is close.

Heuristic only: there's no dynamics/simulator here. It flags "hazard frames"
where front lidar is below a threshold and measures if the predicted steering
turns toward the side with more clearance.
"""

from __future__ import annotations

import argparse
import os
import json

import mlx.core as mx
import mlx.nn as nn

from ncps import IcraCfCCell
from datasets.icra2020_lidar_collision_avoidance import load_data


def build_cell(total_input_dim: int) -> IcraCfCCell:
    return IcraCfCCell(lidar_bins=total_input_dim - 2, state_dim=2, profile="cfc_icra")


def evaluate_sequence(
    cell: IcraCfCCell,
    packed_seq: mx.array,  # [T, L+2]
    hx: mx.array | None = None,
    *,
    front_thresh: float = 1.2,
    turn_deadband: float = 0.05,
) -> dict:
    L = cell.lidar_bins
    T = int(packed_seq.shape[0])
    # Heuristic sectors
    mid = L // 2
    front_lo = max(0, mid - 30)
    front_hi = min(L, mid + 30)

    safe = mx.array(0, dtype=mx.int32)
    hazard = mx.array(0, dtype=mx.int32)
    neutral = mx.array(0, dtype=mx.int32)

    # Stream over time
    if hx is None:
        hx = cell.zero_state(1)

    for t in range(T):
        row = packed_seq[t]
        lidar_t = row[:L]
        state_t = row[L:]
        # One-step inference
        y_t, hx = cell.step(mx.expand_dims(state_t, 0), mx.expand_dims(lidar_t, 0), hx=hx)

        # Heuristic: evaluate only if front is too close
        front_min = mx.min(lidar_t[front_lo:front_hi])
        if front_min > mx.array(front_thresh, dtype=mx.float32):
            neutral = neutral + 1
            continue

        # Which side is safer? Compare mean clearance left vs right
        left_clear = mx.mean(lidar_t[:mid])
        right_clear = mx.mean(lidar_t[mid:])
        steer = y_t[0, 0]

        # Deadband: treat small |steer| as indecision (neutral)
        if mx.abs(steer) < mx.array(turn_deadband, dtype=mx.float32):
            neutral = neutral + 1
            continue

        # Safe if turning toward the clearer side
        safer_left = left_clear > right_clear
        turning_left = steer < 0.0
        safe = safe + mx.where(safer_left == turning_left, 1, 0)
        hazard = hazard + mx.where(safer_left != turning_left, 1, 0)

    return {
        "safe": safe,
        "hazard": hazard,
        "neutral": neutral,
        "frames": mx.array(T, dtype=mx.int32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="artifacts/icra_cfc/weights.npz")
    ap.add_argument("--hx", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--config", type=str, default="artifacts/icra_cfc/config.json")
    ap.add_argument("--front_thresh", type=float, default=1.2)
    ap.add_argument("--turn_deadband", type=float, default=0.05)
    args = ap.parse_args()

    # Load data and pick a sequence
    (train_x, _), (test_x, _) = load_data(seq_len=args.seq_len)
    batch = test_x if test_x.shape[0] > 0 else train_x
    x = batch[min(args.index, int(batch.shape[0]) - 1)]  # [T, L+2, 1]
    x = mx.reshape(x, (x.shape[0], -1))

    # Build cell and load weights
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        total_dim = int(cfg.get("lidar_bins", x.shape[-1] - 2)) + int(cfg.get("state_dim", 2))
    else:
        total_dim = int(x.shape[-1])

    cell = build_cell(total_dim)
    if os.path.isfile(args.weights):
        cell.load_weights(args.weights)

    hx = None
    if args.hx and os.path.isfile(args.hx):
        hx = mx.load(args.hx)["hx"]

    metrics = evaluate_sequence(
        cell,
        x,
        hx=hx,
        front_thresh=args.front_thresh,
        turn_deadband=args.turn_deadband,
    )
    print("Evaluation (heuristic):", metrics)


if __name__ == "__main__":
    main()
