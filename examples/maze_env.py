"""Headless tile-map environment with simple LIDAR and expert policy.

Provides: MAP_ASCII, TILE, is_wall, raycast, and a simulator that can
generate (lidar,state,steer) sequences for training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import mlx.core as mx


MAP_ASCII = [
    "############################################",
    "#......................#...................#",
    "#..###########..####...#..#######..#####..#",
    "#..#.........#..#..#...#..#.....#..#...#..#",
    "#..#..#####..#..#..#......#..#..#..#.#.#..#",
    "#..#..#...#..#..#..######.#..#..#..#.#.#..#",
    "#..#..#...#..#..#........#.#..#..#..#.#.#..#",
    "#..#..#...#..#..########.#.#..#..#..#.#.#..#",
    "#..#..#...#..#...........#.#..#..#..#.#.#..#",
    "#..#..#####..#############.#..####..#.#.#..#",
    "#..#.................................#.#.#..#",
    "#..###############################...#.#.#..#",
    "#.......................................#...#",
    "############################################",
]

TILE = 40  # px per cell


def is_wall(px: float, py: float) -> bool:
    gx = int(px // TILE)
    gy = int(py // TILE)
    if gx < 0 or gy < 0 or gy >= len(MAP_ASCII) or gx >= len(MAP_ASCII[0]):
        return True
    return MAP_ASCII[gy][gx] == "#"


def raycast(px: float, py: float, ang: float, max_dist: float) -> float:
    step = 4.0
    dist = 0.0
    while dist < max_dist:
        x = px + dist * math.cos(ang)
        y = py + dist * math.sin(ang)
        if is_wall(x, y):
            break
        dist += step
    return min(dist, max_dist)


@dataclass
class SimConfig:
    seq_len: int = 64
    bins: int = 181
    fov: float = math.radians(270)
    lidar_max: float = TILE * 12
    speed: float = 120.0  # px/s
    dt: float = 0.1       # s per step (10 Hz)
    steer_gain: float = 1.0


def expert_steer(dists: List[float]) -> float:
    mid = len(dists) // 2
    left = sum(dists[:mid]) / max(1, mid)
    right = sum(dists[mid:]) / max(1, len(dists) - mid)
    front = min(dists[mid - 4: mid + 5])
    cmd = -0.8 if left < right else 0.8
    if front > TILE * 4:
        cmd *= 0.3
    return cmd


def simulate_episode(cfg: SimConfig, start: Tuple[float, float, float] | None = None) -> Tuple[mx.array, mx.array]:
    # State
    if start is None:
        rx, ry, heading = TILE * 2.5, TILE * (len(MAP_ASCII) - 2.5), -math.pi / 2
    else:
        rx, ry, heading = start

    ANG0 = -cfg.fov / 2
    inputs: List[List[float]] = []
    targets: List[float] = []

    for _ in range(cfg.seq_len):
        # LIDAR
        dists: List[float] = []
        for i in range(cfg.bins):
            a = heading + ANG0 + cfg.fov * (i / (cfg.bins - 1))
            d = raycast(rx, ry, a, cfg.lidar_max)
            dists.append(d)

        steer = expert_steer(dists)
        # Inputs: normalize distances and add [v, 0] as state
        norm = [min(1.0, d / cfg.lidar_max) for d in dists]
        v = cfg.speed / (TILE * 4)  # rough normalization factor
        inputs.append(norm + [v, 0.0])
        targets.append(steer)

        # Integrate
        heading += cfg.steer_gain * steer * cfg.dt
        nx = rx + math.cos(heading) * cfg.speed * cfg.dt
        ny = ry + math.sin(heading) * cfg.speed * cfg.dt
        if not is_wall(nx, ny):
            rx, ry = nx, ny

    x = mx.array(inputs, dtype=mx.float32)
    y = mx.array([[t] for t in targets], dtype=mx.float32)
    return x, y


def make_batch(cfg: SimConfig, episodes: int) -> Tuple[mx.array, mx.array]:
    xs, ys = [], []
    for _ in range(episodes):
        x, y = simulate_episode(cfg)
        xs.append(mx.expand_dims(x, 0))
        ys.append(mx.expand_dims(y, 0))
    X = mx.concatenate(xs, axis=0)
    Y = mx.concatenate(ys, axis=0)
    return X, Y

