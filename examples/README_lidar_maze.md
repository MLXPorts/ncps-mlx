Lidar Robot Maze — NCPS + MLX

This example set demonstrates an NCPS‑style controller for a LIDAR robot in a tile‑maze and a real‑data viewer for the ICRA LIDAR logs, implemented on Apple MLX. It includes:

- An imitation trainer that uses a wired CfC with a single explicit motor neuron (NCPS wiring) on a simulated maze with ray‑cast LIDAR.
- A PPO reinforcement‑learning trainer whose actor is also a wired CfC motor neuron policy.
- Two Pygame demos: a real‑time maze navigation viewer and an ICRA LIDAR viewer.
- NCPS wiring presets that expose sensory and motor neurons, so inputs flow through explicit sensory synapses and the action is read from a motor neuron.

All computation stays on device tensors (MLX arrays): no NumPy conversions, no `.item()`, no CPU round‑trips in library code.


## Files and entry points

- examples/maze_env.py — Tile‑maze + LIDAR simulator and a tiny “expert” for imitation.
- examples/wiring_presets.py — Helper to build NCPS sensory/motor wirings.
- examples/maze_train_mlx.py — Imitation training on the maze (wired CfC motor policy). Saves to artifacts/maze_cfc/.
- examples/maze_rl_ppo_mlx.py — PPO trainer with a wired CfC motor policy and a CfC critic. Saves to artifacts/maze_ppo/.
- examples/maze_nav_pygame.py — Visualizes the maze and drives the robot: manual, heuristic autopilot, or NCPS model (press `M`).
- examples/icra_lidar_pygame.py — Real ICRA LIDAR replay with a time‑based “virtual robot” view.
- examples/icra_lidar_mlx.py — ICRA CfC imitation trainer that auto‑saves weights/hx/config.
- examples/icra_lidar_infer.py — Batch inference helper on ICRA.

### ICRA specifics

- Data: auto‑downloads a pinned packed NPZ (icra_lds) with checksum verification into `datasets/`. No manual steps needed.
- Training: deterministic by construction — fixed splits, seeded batching, per‑epoch validation + best checkpoint.
- Artifacts: `artifacts/icra_cfc/weights.npz`, `hx.npz`, `config.json`, and `training_log.csv` (epoch, train_loss, train_mae, val_loss, val_mae).


## Quick start

- Train a maze imitation model (fast):
  - `PYTHONPATH=. python examples/maze_train_mlx.py`
  - Writes `artifacts/maze_cfc/weights.npz` and `config.json`.

- Train a maze RL policy (PPO):
  - `PYTHONPATH=. python examples/maze_rl_ppo_mlx.py`
  - Writes `artifacts/maze_ppo/weights.npz`.

- Run the maze viewer:
  - `PYTHONPATH=. python examples/maze_nav_pygame.py`
  - Controls: `A` autopilot toggle, `M` model‑drive toggle (loads `maze_cfc` weights on first press), `L` rays on/off, arrows (manual mode), `+/-` speed, `[ ]` steering gain.

- Run the ICRA viewer (needs pygame):
  - `PYTHONPATH=. python examples/icra_lidar_pygame.py`


## How the NCPS wiring is used here

- Motor neuron (actor’s output):
  - We create a CfC with `units = wiring` where `wiring = wirings.FullyConnected(units=H, output_dim=1)` (or the provided `make_sensory_motor_wiring`). The wiring’s motor size is 1; the CfC reads the motor neuron directly as the steering command (after `tanh`).

- Sensory neurons (inputs → hidden):
  - `make_sensory_motor_wiring(input_dim, units, output_dim=1, sectors=3)` partitions the input channels into angular sectors (e.g., left/middle/right) and connects each group to a disjoint subset of hidden neurons via the sensory adjacency matrix. That makes sensory→inter synapses explicit in the NCPS graph.

- Interneurons (recurrent core):
  - The remainder of the units are recurrent (self‑connections can be enabled) and learn to modulate the motor.

This reproduces the NCPS flavor: inputs flow through sensory synapses into an interpretable recurrent core; actions are read from a dedicated motor neuron.


## Control‑theory context from the paper (why NCPS)

The original Neuronal Circuit Policies (NCP) paper re‑purposes the biological tap‑withdrawal (TW) circuit of C. elegans for control tasks using an interpretable ODE‑based model:

- Neuron dynamics (single‑compartment membrane ODE):
  - `Cm dvi/dt = GLeak (VLeak − vi) + Σ Iin` (graded, non‑spiking neurons).
- Chemical synapses: `g(Vpre) = w / (1 + e^{σ(Vpre+μ)})`, current `Is = g(Vpre)(E − Vpost)`; sign via reversal potential E.
- Gap junctions: `Î_{i,j} = ω̂ (vj − vi)` (bidirectional ohmic coupling).
- Sensory and motor “pairs”:
  - Sensory: two neurons per scalar input (Sp for positive, Sn for negative) via piecewise linear maps into [−70 mV, −20 mV].
  - Motor: two neurons (Mp/Mn) mapped back to actuator ranges (ymin..ymax) via piecewise linear maps.
- Learning: search‑based RL (Random Search + Objective Estimate) filters rollouts to handle stochastic returns (Algorithm 1 in paper).
- Interpretability: antagonistic command neurons (e.g., AVA/AVB) exhibit competitive dynamics; neuron‑level traces are meaningful.

Our current implementation is CfC‑based (closed‑form continuous‑time) with NCPS wiring for sensory/motor structure. It preserves the NCPS interpretability (sensory→inter→motor graph with explicit motor neurons), and we provide both imitation and PPO learning. As a next step (see Roadmap) we plan a faithful TW cell with ODE synapses and search‑based RL to match the paper’s solver and learning loop.


## What’s implemented vs planned

Implemented now (MLX‑native):
- Imitation on the maze with a wired CfC motor neuron (NCPS wiring) — examples/maze_train_mlx.py.
- PPO RL on the maze with a wired CfC actor (motor neuron) and CfC critic — examples/maze_rl_ppo_mlx.py.
- NCPS wiring presets that expose sensory and motor nodes — examples/wiring_presets.py.
- Real‑time visualizations (maze + ICRA), with time‑based stepping and device‑side tensors only.

Planned (paper‑faithful TW):
- Add a TWCell (ODE membrane + chemical/gap synapses, hybrid implicit/explicit step) and the search‑based RL loop (random‑search + objective estimate filtering).
- Add paper’s sensory/motor pair mappings (Sp/Sn and Mp/Mn) as first‑class modules and an ego‑view ring buffer to the ICRA viewer.


## How the code works (training and control loops)

- Imitation (maze):
  - Generate expert demos: examples/maze_env.py simulates episodes with a simple sector‑based avoidance rule.
  - Build NCPS wiring: `make_sensory_motor_wiring(input_dim=bins+2, units=64, output_dim=1)`.
  - Model: CfC(input_size=bins+2, units=wiring, proj_size=None, return_sequences=True). The readout is the 1‑neuron motor channel.
  - Train: MSE between predicted steer and expert steer using MLX’s `nn.value_and_grad`. All tensors are MLX arrays; no `.item()`.

- PPO (maze):
  - Actor: wired CfC motor neuron; Critic: compact CfC projecting to value.
  - Rollout: single‑env, on‑policy; compute log‑probs under Gaussian policy (mean from motor neuron, log‑std learned); collect (obs, act, logp, val, rew, done).
  - GAE: compute advantages and returns (device ops), normalize advantages.
  - Update: clipped surrogate objective; value and entropy terms; minibatch SGD, MLX optimizer.

- Viewer (maze):
  - Pygame draws a tiled map and robot pose; LIDAR rays are optional; stepping is wall‑clock based; model drive uses the motor neuron if toggled.

- Viewer (ICRA):
  - Replays real LIDAR sequences; movement is time‑based; you can disable accumulation to avoid moiré. No odometry is present in the logs, so scans are ego‑centric.


## Running details and tips

- Device: MLX automatically uses GPU/ANE if available; we do not convert to host floats in library code.
- Speed: For PPO, raise `steps_per_epoch` and `updates` in examples/maze_rl_ppo_mlx.py for better policies. Use multiple envs longer‑term.
- Stability: Tweak steering gain and smoothing in viewers with `[ ]` and `+/-`.


## Roadmap and validation checklist

- [x] NCPS motor neuron used in imitation and PPO actor.
- [x] Explicit sensory wiring via sensory adjacency (sectors → hidden groups).
- [x] MLX‑only tensors; no `.item()` or `float(mx.array)` in library.
- [ ] Sensory pair mapping Sp/Sn (paper Eqs. 2–3) and motor pair Mp/Mn (Eqs. 4–5) helpers threaded through inputs/outputs.
- [ ] TWCell (ODE with chemical/gap synapses) + search‑based RL loop.
- [ ] ICRA viewer ego‑ring and optional odometry support if pose logs are available.


## References

- Mathias Lechner, Ramin M. Hasani, Radu Grosu. “Neuronal Circuit Policies.” arXiv:1803.08554.
- Wicks & Rankin (1995), Wicks et al. (1996) on TW circuit dynamics.
