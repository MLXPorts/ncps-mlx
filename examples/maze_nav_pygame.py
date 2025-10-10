"""Tile-map navigation demo with simulated LIDAR (Pygame).

This provides an actual environment (walls + corridors) and a simple robot
that moves around in real time. LIDAR rays can be toggled on/off; control can
be manual (keyboard), a tiny avoidance autopilot, or a trained liquid neural network.

Usage:
  PYTHONPATH=. python examples/maze_nav_pygame.py

Controls:
  Arrow keys: steer (left/right) and throttle up/down (manual mode)
  A:          toggle autopilot (simple obstacle avoidance)
  M:          toggle liquid neuron model drive (requires trained weights)
  L:          toggle LIDAR ray visualization
  C:          clear the path trace
  +/-:        speed down/up
  [ / ]:      steering gain down/up
  Q / Esc:    quit

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
    pygame.display.set_caption("Maze Navigation Demo (Pygame)")
    clock = pygame.time.Clock()

    # Robot state
    rx, ry = TILE * 2.5, TILE * (len(MAP_ASCII) - 2.5)
    heading = -math.pi / 2
    speed = 120.0  # px/s
    steer_gain = 1.2
    show_rays = True
    autopilot = True
    model_drive = False
    model = None
    hx = None
    trace = []  # path points

    # LIDAR parameters
    FOV = math.radians(270)
    BINS = 181
    ANG0 = -FOV / 2
    MAX_R = TILE * 12

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
                if event.key == pygame.K_a:
                    autopilot = not autopilot
                if event.key == pygame.K_c:
                    trace.clear()
                if event.key == pygame.K_m and CfC is not None:
                    # Toggle model drive; lazy-init wired CfC with motor neuron and load weights if present
                    if model is None:
                        from examples.wiring_presets import make_sensory_motor_wiring
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
                        import os
                        w = os.path.join("artifacts", "maze_cfc", "weights.npz")
                        if os.path.isfile(w):
                            model.load_weights(w)
                            print("✓ Loaded trained CfC weights for liquid neuron navigation")
                        else:
                            print("⚠ Warning: No trained weights found, using random initialization")
                            print("  Train the model first: python -m examples.maze_train_mlx")
                        hx = mx.zeros((1, 64), dtype=mx.float32)
                    model_drive = not model_drive
                    mode_str = "liquid neurons" if model_drive else "autopilot/manual"
                    print(f"🧠 Control mode: {mode_str}")
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

        # Controller
        steer_cmd = 0.0
        throttle = 1.0
        keys = pygame.key.get_pressed()
        if model_drive and CfC is not None and model is not None and hx is not None:
            norm = [min(1.0, d / MAX_R) for d in dists]
            v = speed / (TILE * 4)
            x = mx.array([[norm + [v, 0.0]]], dtype=mx.float32)
            mu, hx = model(x, hx=hx)
            steer_cmd = float(mu[0, 0, 0].tolist())
        elif not autopilot:
            # Manual: arrows
            if keys[pygame.K_LEFT]:
                steer_cmd -= 1.0
            if keys[pygame.K_RIGHT]:
                steer_cmd += 1.0
            if keys[pygame.K_UP]:
                throttle = 1.0
            elif keys[pygame.K_DOWN]:
                throttle = 0.4
        else:
            # Tiny avoidance: compare left/right sectors + front clearance
            mid = BINS // 2
            left = sum(dists[:mid]) / mid
            right = sum(dists[mid:]) / mid
            front = min(dists[mid - 5: mid + 5])
            steer_cmd = -0.8 if left < right else 0.8
            if front > TILE * 4:
                steer_cmd *= 0.3

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
        txt = f"mode={'AUTO' if autopilot else 'MAN'} v={speed:.0f}px/s gain={steer_gain:.2f} rays={'on' if show_rays else 'off'}"
        screen.blit(font.render(txt, True, (230, 230, 230)), (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    main()
