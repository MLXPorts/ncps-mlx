"""ICRA LIDAR visual demo with streaming inference (Pygame).

This replays a real ICRA sequence and renders LIDAR rays around a vehicle
while showing the model's predicted steering (and the ground truth).

Requirements:
  - pygame (pip install pygame)

Usage:
  PYTHONPATH=. python examples/icra_lidar_pygame.py

Controls:
  - Space: pause/resume
  - N:     step one frame
  - R:     reset hidden state
  - Q/ESC: quit
"""

from __future__ import annotations

import os
import json
import math
import time
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

try:
    import pygame
except Exception as e:  # pragma: no cover
    raise SystemExit("Pygame is required: pip install pygame") from e

from ncps import IcraCfCCell
from ncps.ncps_mlx.datasets.icra2020_lidar_collision_avoidance import load_data


ARTIFACTS = "artifacts/icra_cfc"
CFG_PATH = os.path.join(ARTIFACTS, "config.json")
WEIGHTS_PATH = os.path.join(ARTIFACTS, "weights.npz")
HX_PATH = os.path.join(ARTIFACTS, "hx.npz")


def build_model(total_input_dim: int) -> IcraCfCCell:
    return IcraCfCCell(lidar_bins=total_input_dim - 2, state_dim=2, profile="cfc_icra")


def select_sequence(seq_len: int, index: int = 0) -> Tuple[mx.array, mx.array]:
    (_, _), (test_x, test_y) = load_data(seq_len=seq_len)
    batch = test_x if test_x.shape[0] > 0 else load_data(seq_len=seq_len)[0][0]
    x = batch[min(index, int(batch.shape[0]) - 1)]  # [T, L+2, 1]
    x = mx.reshape(x, (x.shape[0], -1))             # [T, L+2]
    return x, test_y[min(index, int(test_y.shape[0]) - 1)] if test_y.shape[0] > 0 else mx.zeros((x.shape[0], 1), dtype=mx.float32)


def main() -> None:  # pragma: no cover
    # Load config or infer dims from data
    seq_len = 64
    x_seq, y_seq = select_sequence(seq_len=seq_len, index=0)

    if os.path.isfile(CFG_PATH):
        cfg = json.load(open(CFG_PATH, "r", encoding="utf-8"))
        lidar_bins = int(cfg.get("lidar_bins", x_seq.shape[-1] - 2))
        state_dim = int(cfg.get("state_dim", 2))
    else:
        lidar_bins = x_seq.shape[-1] - 2
        state_dim = 2

    # Build model and load weights if present
    cell = build_model(lidar_bins + state_dim)
    if os.path.isfile(WEIGHTS_PATH):
        cell.load_weights(WEIGHTS_PATH)

    hx = cell.zero_state(1)
    if os.path.isfile(HX_PATH):
        try:
            hx = mx.load(HX_PATH)["hx"]
        except Exception:
            pass

    # Pygame setup
    pygame.init()
    W, H = 900, 900
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("ICRA LIDAR Demo (MLX)")
    clock = pygame.time.Clock()

    # World/map surface for a "virtual maze" feel (accumulates points/segments)
    MAP_W, MAP_H = 2400, 2400
    map_surf = pygame.Surface((MAP_W, MAP_H))
    map_surf.fill((10, 10, 10))

    # Initial robot pose in world
    rx, ry = MAP_W // 2, MAP_H // 2
    heading = -math.pi / 2  # facing upward
    # Real-time control parameters
    DATA_HZ = 10.0          # play dataset at ~10 Hz (approx. sensor rate)
    speed_mps = 0.6         # nominal forward speed in "meters" per second (scaled)
    dist_scale = 60.0       # px per meter (also used for lidar ray length)
    speed_px = speed_mps * dist_scale
    steer_gain = 0.8        # radians per unit steer
    use_ground_truth = False

    # Camera projection (center on robot)
    def to_screen(wx: float, wy: float) -> Tuple[int, int]:
        cx, cy = W // 2, H // 2
        return int(wx - rx + cx), int(wy - ry + cy)

    # Lidar rendering settings
    rays_step = max(1, lidar_bins // 180)
    angle_lo, angle_hi = -math.pi * 0.75, math.pi * 0.75

    # Steering smoothing
    steer_smooth = 0.0
    alpha = 0.15

    paused = False
    t = 0
    fps_cap = 30  # display FPS cap (independent of data rate)
    data_accum = 0.0
    data_dt = 1.0 / DATA_HZ
    draw_segments = True
    accumulate_map = True

    def draw_frame(t_idx: int, hx_in: mx.array) -> Tuple[mx.array, float]:
        nonlocal rx, ry, heading, steer_smooth
        lidar_t = x_seq[t_idx, :lidar_bins]
        state_t = x_seq[t_idx, lidar_bins:]
        # Model step
        y_t, hx_out = cell.step(mx.expand_dims(state_t, 0), mx.expand_dims(lidar_t, 0), hx=hx_in)

        # Predicted or GT steering
        steer = float(y_t[0, 0].tolist())
        # Smooth steering for readability
        steer_smooth = (1.0 - alpha) * steer_smooth + alpha * steer

        # Integrate a simple egocentric motion model
        # (Motion integration now happens in the main loop with wall-clock dt)

        # Paint LIDAR endpoints onto a world map
        lidar = lidar_t.tolist()
        # Draw rays and optionally connect adjacent endpoints to suggest walls
        pts = []
        for i in range(0, lidar_bins, rays_step):
            a_rel = angle_lo + (angle_hi - angle_lo) * (i / (lidar_bins - 1))
            a = heading + a_rel
            dist = max(0.2, min(2.5, float(lidar[i])))
            wx = rx + dist * dist_scale * math.cos(a)
            wy = ry + dist * dist_scale * math.sin(a)
            pts.append((int(wx), int(wy)))
            if accumulate_map:
                pygame.draw.circle(map_surf, (70, 170, 130), (int(wx), int(wy)), 1)
        if draw_segments and len(pts) > 1:
            pygame.draw.lines(map_surf, (50, 120, 100), False, pts, 1)

        # Compose view centered on robot
        screen.fill((15, 15, 18))
        view = pygame.Rect(0, 0, W, H)
        view.center = (int(rx), int(ry))
        screen.blit(map_surf, (0, 0), view)

        # Draw current rays atop
        cx, cy = to_screen(rx, ry)
        for i in range(0, lidar_bins, rays_step * 6):
            a_rel = angle_lo + (angle_hi - angle_lo) * (i / (lidar_bins - 1))
            a = heading + a_rel
            dist = min(2.0, float(lidar[i]))
            ex = cx + int(dist * dist_scale * math.cos(a))
            ey = cy + int(dist * dist_scale * math.sin(a))
            pygame.draw.line(screen, (80, 200, 120), (cx, cy), (ex, ey), 1)

        # Robot marker and steering vector
        pygame.draw.circle(screen, (230, 230, 230), (cx, cy), 6)
        arrow_len = int(80 * min(1.0, abs(steer_smooth)))
        ax = cx + int(arrow_len * math.cos(heading + steer_smooth * 0.6))
        ay = cy + int(arrow_len * math.sin(heading + steer_smooth * 0.6))
        pygame.draw.line(screen, (220, 140, 90), (cx, cy), (ax, ay), 3)

        # HUD
        font = pygame.font.SysFont("monospace", 16)
        hud = font.render(
            f"t={t_idx+1}/{int(x_seq.shape[0])} steer={steer:+.3f} v={speed_mps:.2f}m/s data={DATA_HZ:.0f}Hz",
            True,
            (230, 230, 230),
        )
        screen.blit(hud, (10, 10))
        pygame.display.flip()
        return hx_out, steer_smooth

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_n:
                    paused = True
                    hx, _ = draw_frame(t, hx)
                    t = (t + 1) % int(x_seq.shape[0])
                if event.key == pygame.K_r:
                    hx = cell.zero_state(1)
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed_mps = min(2.0, speed_mps + 0.1)
                    speed_px = speed_mps * dist_scale
                if event.key == pygame.K_MINUS:
                    speed_mps = max(0.1, speed_mps - 0.1)
                    speed_px = speed_mps * dist_scale
                if event.key == pygame.K_c:
                    map_surf.fill((10, 10, 10))
                if event.key == pygame.K_b:
                    draw_segments = not draw_segments
                if event.key == pygame.K_m:
                    accumulate_map = not accumulate_map
                if event.key == pygame.K_f:
                    # toggle data rate 5/10/20 Hz
                    DATA_HZ = 20.0 if DATA_HZ < 10 else (10.0 if DATA_HZ > 10 else 5.0)
                    data_dt = 1.0 / DATA_HZ

        # Wall-clock step: integrate motion with dt, advance data at DATA_HZ
        dt = clock.tick(fps_cap) / 1000.0
        if not paused:
            # Integrate motion each frame (based on smoothed steer from last data tick)
            heading += steer_smooth * steer_gain * dt
            rx += speed_px * math.cos(heading) * dt
            ry += speed_px * math.sin(heading) * dt

            # Accumulate and advance data frames at target rate
            data_accum += dt
            while data_accum >= data_dt:
                hx, _ = draw_frame(t, hx)
                t = (t + 1) % int(x_seq.shape[0])
                data_accum -= data_dt
        else:
            # Even when paused, keep refreshing display (no motion)
            pygame.display.flip()


if __name__ == "__main__":  # pragma: no cover
    main()
