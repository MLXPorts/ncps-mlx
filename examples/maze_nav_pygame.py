"""Tile-map navigation demo with liquid neural network control (Pygame).

This demonstrates a robot navigating a maze environment using a trained
Closed-form Continuous-time (CfC) liquid neural network from the ICRA 2020
paper "Gershgorin Loss Stabilizes the Recurrent Neural Network Compartment
of an End-To-End Robot Learning Scheme".

The network was trained on real robot demonstrations with LIDAR sensor data
and outputs steering commands in real-time. It uses a Conv1D backbone to
process LIDAR readings and a CfC cell for temporal dynamics.

Architecture:
  - Conv1D(12, k=5, s=3) → Conv1D(16, k=5, s=3) → Conv1D(24, k=5, s=2) 
    → Conv1D(1, k=1, s=1) → Flatten
  - CfC with 64 units
  - Output: steering command

Usage:
  PYTHONPATH=. python examples/maze_nav_pygame.py

Controls:
  Arrow keys: Manual override (left/right to steer)
  L:          Toggle LIDAR ray visualization
  C:          Clear the path trace
  +/-:        Speed down/up
  [ / ]:      Steering gain down/up
  Q / Esc:    Quit

Training:
  To train the ICRA model on real robot demonstrations:
    python examples/icra_lidar_mlx.py
  This will create weights in artifacts/icra_cfc/
"""

from __future__ import annotations

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

try:
    import pygame
except Exception as e:  # pragma: no cover
    raise SystemExit("Pygame is required: pip install pygame") from e

try:
    # When run as a script
    from examples.maze_env import MAP_ASCII, TILE, is_wall, raycast
except Exception:
    # When run as module
    from .maze_env import MAP_ASCII, TILE, is_wall, raycast

W = TILE * len(MAP_ASCII[0])
H = TILE * len(MAP_ASCII)


def main() -> None:  # pragma: no cover
    pygame.init()
    screen = pygame.display.set_mode((min(W, 1200), min(H, 900)))
    pygame.display.set_caption("Liquid Neural Network Maze Navigation (ICRA)")
    clock = pygame.time.Clock()

    # Initialize liquid neural network from ICRA 2020 paper
    # Uses Conv1D backbone + CfC trained on real robot demonstrations
    print("Initializing ICRA liquid neural network...")
    from ncps import IcraCfCCell
    import os
    import json
    
    # Load config to get correct LIDAR dimensions
    config_path = os.path.join("artifacts", "icra_cfc", "config.json")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        ICRA_BINS = config['lidar_bins']
        STATE_DIM = config['state_dim']
        print(f"  Config: {ICRA_BINS} LIDAR bins, {STATE_DIM} state dims")
    else:
        # Fallback defaults from ICRA dataset
        ICRA_BINS = 538
        STATE_DIM = 2
        print(f"  Using default: {ICRA_BINS} LIDAR bins, {STATE_DIM} state dims")
    
    # Simulation LIDAR resolution
    SIM_BINS = 181
    
    # Build ICRA CfC cell with Conv1D backbone
    model = IcraCfCCell(lidar_bins=ICRA_BINS, state_dim=STATE_DIM, profile="cfc_icra")
    
    # Load trained weights from ICRA obstacle avoidance dataset
    weights_path = os.path.join("artifacts", "icra_cfc", "weights.npz")
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print("✓ Loaded ICRA trained weights (real robot demonstrations)")
    else:
        print("⚠ Warning: No ICRA weights found!")
        print("  Train first: python examples/icra_lidar_mlx.py")
        print("  Continuing with random initialization...")
    
    hx = model.zero_state(batch_size=1)

    # Robot state
    rx, ry = TILE * 2.5, TILE * (len(MAP_ASCII) - 2.5)
    heading = -math.pi / 2
    speed = 120.0  # px/s
    steer_gain = 1.2
    show_rays = True
    manual_override = False
    trace = []  # path points

    # LIDAR parameters for simulation
    FOV = math.radians(270)
    MAX_R = TILE * 12
    ANG0 = -FOV / 2

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_l:
                    show_rays = not show_rays
                if event.key == pygame.K_c:
                    trace.clear()
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(300.0, speed + 20.0)
                if event.key == pygame.K_MINUS:
                    speed = max(40.0, speed - 20.0)
                if event.key == pygame.K_LEFTBRACKET:
                    steer_gain = max(0.4, steer_gain - 0.1)
                if event.key == pygame.K_RIGHTBRACKET:
                    steer_gain = min(2.5, steer_gain + 0.1)

        # Simulate LIDAR in robot frame (at simulation resolution)
        dists: List[float] = []
        for i in range(SIM_BINS):
            a = heading + ANG0 + FOV * (i / (SIM_BINS - 1))
            d = raycast(rx, ry, a, MAX_R)
            dists.append(d)

        # Controller: Liquid neural network with manual override
        keys = pygame.key.get_pressed()
        manual_override = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
        
        if manual_override:
            # Manual steering override
            steer_cmd = 0.0
            if keys[pygame.K_LEFT]:
                steer_cmd -= 1.0
            if keys[pygame.K_RIGHT]:
                steer_cmd += 1.0
        else:
            # Liquid neural network control (ICRA CfC)
            # Normalize LIDAR readings [0, 1]
            lidar_norm_list = [min(1.0, d / MAX_R) for d in dists]
            
            # Interpolate from SIM_BINS to ICRA_BINS using numpy
            if SIM_BINS != ICRA_BINS:
                import numpy as np
                lidar_np = np.array(lidar_norm_list, dtype=np.float32)
                x_sim = np.linspace(0, 1, SIM_BINS)
                x_icra = np.linspace(0, 1, ICRA_BINS)
                lidar_interp = np.interp(x_icra, x_sim, lidar_np)
                lidar_norm = mx.array([[lidar_interp.tolist()]], dtype=mx.float32)
            else:
                lidar_norm = mx.array([[lidar_norm_list]], dtype=mx.float32)
            
            # Vehicle state: [velocity, angular_velocity] normalized
            v_norm = speed / (TILE * 4)  # Normalize velocity
            w_norm = 0.0  # Angular velocity (not used in this simple sim)
            state = mx.array([[v_norm, w_norm]], dtype=mx.float32)
            
            # Expand to time dimension [batch=1, time=1, features]
            lidar_seq = mx.expand_dims(lidar_norm, axis=1)
            state_seq = mx.expand_dims(state, axis=1)
            
            # Forward pass through ICRA CfC (returns [batch, time, 1])
            mu, hx = model(state_seq, lidar_seq, hx=hx, return_state=True)
            steer_cmd = float(mu[0, 0, 0].item())
        
        throttle = 1.0

        # Integrate motion
        heading += steer_gain * steer_cmd * dt
        vx = math.cos(heading) * speed * throttle
        vy = math.sin(heading) * speed * throttle
        next_rx, next_ry = rx + vx * dt, ry + vy * dt
        # If collision with a wall, stop forward motion but allow turning
        if not is_wall(next_rx, next_ry):
            rx, ry = next_rx, next_ry

        trace.append((int(rx), int(ry)))
        if len(trace) > 2000:
            trace = trace[-2000:]

        # Draw
        screen.fill((18, 18, 22))
        # Map
        for gy, row in enumerate(MAP_ASCII):
            for gx, ch in enumerate(row):
                if ch == '#':
                    pygame.draw.rect(
                        screen,
                        (40, 40, 50),
                        pygame.Rect(gx * TILE, gy * TILE, TILE, TILE),
                    )
        # Rays
        if show_rays:
            for i, d in enumerate(dists[::4]):  # draw fewer for speed
                a = heading + ANG0 + FOV * ((i * 4) / (BINS - 1))
                ex = rx + d * math.cos(a)
                ey = ry + d * math.sin(a)
                pygame.draw.aaline(screen, (90, 200, 140), (rx, ry), (ex, ey))

        # Path trace
        if len(trace) > 1:
            pygame.draw.lines(screen, (120, 120, 180), False, trace, 2)

        # Robot
        pygame.draw.circle(screen, (230, 230, 230), (int(rx), int(ry)), 6)
        nose = (int(rx + 14 * math.cos(heading)), int(ry + 14 * math.sin(heading)))
        pygame.draw.line(screen, (230, 150, 90), (int(rx), int(ry)), nose, 3)

        # HUD
        font = pygame.font.SysFont("monospace", 16)
        mode_txt = "MANUAL" if manual_override else "LIQUID NEURONS"
        txt = f"{mode_txt} | speed={speed:.0f}px/s | gain={steer_gain:.2f} | rays={'on' if show_rays else 'off'}"
        screen.blit(font.render(txt, True, (100, 255, 100)), (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    main()
