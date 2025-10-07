"""Legacy wiring API re-exporting MLX-native implementations."""

from ncps.mlx.wirings import AutoNCP, FullyConnected, NCP, Random, Wiring

__all__ = [
    "Wiring",
    "FullyConnected",
    "Random",
    "NCP",
    "AutoNCP",
]
