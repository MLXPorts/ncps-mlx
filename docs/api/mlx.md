# MLX API Overview

This project exposes MLX-first building blocks for Neural Circuit Policies. The canonical modules are placed under `ncps.mlx`.

## Wiring

```python
from ncps.mlx.wirings import FullyConnected, Random, NCP, AutoNCP

wiring = FullyConnected(units=16, output_dim=8, erev_init_seed=42)
wiring.build(input_dim=10)
```

- `Wiring` is an `mlx.nn.Module` storing `adjacency_matrix` and `sensory_adjacency_matrix` as MLX tensors.
- Random sampling uses MLX’s PRNG keys (`mx.random.key`, `mx.random.split`).
- `synapse_count` / `sensory_synapse_count` return Python ints without materialising NumPy arrays.
- `state_dict()` / `load_state_dict()` serialise MLX-backed wiring state for checkpointing.

## Cells

```python
from ncps.mlx import CfCCell
cell = CfCCell(wiring, mode="default")
```

Cells inherit from `mlx.nn.Module` and lazily create weight matrices based on wiring metadata when first called. They keep everything in MLX tensors.

## Recurrent Layers

```python
from ncps.mlx import CfC
model = CfC(cell, return_sequences=True)
```

`CfC`, `LTC`, `CfCRNN`, `LTCRNN`, and `MMRNN` wrap the cells to deliver sequence processing with MLX modules.

## Training Utilities

Use MLX value/grad + optimisers end-to-end:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

loss_fn = lambda mdl, x, y: mx.mean((mdl(x) - y) ** 2)
value_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = value_and_grad(x, y)
optimizer = optim.Adam(learning_rate=1e-3)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

Persist checkpoints with `model.save_weights("weights.npz")` and bring models back with
`model.load_weights("weights.npz")`; the MLX runtime handles both `.npz` and `.safetensors`.

Consult `docs/guides/mlx_guide.rst` for the full set of MLX usage patterns.

See `examples/mlx_cfc_regression.py`, `examples/mlx_smnist_training.py`, and
`examples/mlx_cheetah_training.py` for end-to-end MLX workflows.

### LTC specifics

`LTCCell` exposes solver controls (`ode_solver` = ``"semi_implicit"``, ``"explicit"``, or
``"runge_kutta"``) and maintains biophysical constraints on conductances and synaptic weights via
`apply_constraints`. Call :meth:`ncps.mlx.LTC.apply_constraints` after optimiser updates to mirror the
original MIT CSAIL training procedure.
