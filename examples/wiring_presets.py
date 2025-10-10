from __future__ import annotations

from typing import List

from ncps import wirings


def make_sensory_motor_wiring(
    input_dim: int,
    units: int,
    *,
    output_dim: int = 1,
    sectors: int = 3,
    self_connections: bool = True,
) -> wirings.Wiring:
    """Construct a single-layer NCPS wiring with explicit sensory and motor neurons.

    - Partitions sensory inputs into ``sectors`` angular groups (e.g., left/mid/right)
      and connects each sector to a disjoint subset of hidden neurons.
    - Declares ``output_dim`` motor neurons (index range [0, output_dim)) per NCPS
      convention. The model will read the motor(s) directly as action(s).
    """
    w = wirings.Wiring(units)
    w.set_output_dim(output_dim)
    w.build(input_dim)

    # Partition hidden neurons into sectors (skip motor indices [0..output_dim-1])
    start = output_dim
    remain = max(0, units - output_dim)
    group_size = max(1, remain // max(1, sectors))
    groups: List[List[int]] = []
    idx = start
    for s in range(sectors):
        end = min(units, idx + group_size)
        if s == sectors - 1:
            end = units
        groups.append(list(range(idx, end)))
        idx = end

    # Map input indices to sector groups (uniform partition over inputs)
    per = max(1, input_dim // max(1, sectors))
    for i in range(input_dim):
        sector = min(sectors - 1, i // per)
        targets = groups[sector] if groups[sector] else []
        for dest in targets:
            # +1 excitatory sensory synapses by default
            w.add_sensory_synapse(i, dest, +1)

    # Optionally add self-connections as +1 to stabilise recurrent identity (not required
    # by the mask logic, but can be useful for explicit wiring inspection).
    if self_connections:
        for n in range(units):
            w.add_synapse(n, n, +1)

    return w

