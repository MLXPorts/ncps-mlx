"""Tile-map navigation demo with liquid neural network control (Pygame).

This demonstrates a robot navigating a maze environment using a trained
Closed-form Continuous-time (CfC) liquid neural network. The network receives
simulated LIDAR sensor data and outputs steering commands in real-time.

The liquid neural network uses sensory-motor wiring with 64 neurons, where
the first neuron acts as a motor neuron controlling steering direction.

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
  To train the liquid neuron model:
    python -m examples.maze_train_mlx
  This will create weights in artifacts/maze_cfc/
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

try:
    from ncps import CfC
except Exception:
    CfC = None  # type: ignore

W = TILE * len(MAP_ASCII[0])
H = TILE * len(MAP_ASCII)


def main() -> None:  # pragma: no cover
    pygame.init()
    screen = pygame.display.set_mode((min(W, 1200), min(H, 900)))
    pygame.display.set_caption("Liquid Neural Network Maze Navigation")
    clock = pygame.time.Clock()

    # Initialize liquid neural network
    print("Initializing liquid neural network...")
    from examples.wiring_presets import make_sensory_motor_wiring
    import os
    
    BINS = 181
    input_dim = BINS + 2
    wiring = make_sensory_motor_wiring(input_dim=input_dim, units=64, output_dim=1)
    model = CfC(
        input_size=input_dim,
        units=wiring,
        proj_size=None,  # Wiring defines motor neuron at index 0
        return_sequences=True,
        batch_first=True,
        mode="default",
        activation="lecun_tanh",
    )
    
    # Load trained weights
    weights_path = os.path.join("artifacts", "maze_cfc", "weights.npz")
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print("✓ Loaded trained liquid neural network weights")
    else:
        print("⚠ Warning: No trained weights found!")
        print("  Train first: python -m examples.maze_train_mlx")
        print("  Continuing with random initialization...")
    
    hx = mx.zeros((1, 64), dtype=mx.float32)

    # Robot state
    rx, ry = TILE * 2.5, TILE * (len(MAP_ASCII) - 2.5)
    heading = -math.pi / 2
    speed = 120.0  # px/s
    steer_gain = 1.2
    show_rays = True
    manual_override = False
    trace = []  # path points

    # LIDAR parameters
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

        # Simulate LIDAR in robot frame
        dists: List[float] = []
        for i in range(BINS):
            a = heading + ANG0 + FOV * (i / (BINS - 1))
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
            # Liquid neural network control
            norm = [min(1.0, d / MAX_R) for d in dists]
            v = speed / (TILE * 4)
            x = mx.array([[norm + [v, 0.0]]], dtype=mx.float32)
            mu, hx = model(x, hx=hx)
            steer_cmd = float(mu[0, 0, 0].tolist())
        
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
