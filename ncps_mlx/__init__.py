from __future__ import annotations

import sys
from importlib import import_module

_pkg = import_module('ncps.ncps_mlx')

__all__ = getattr(_pkg, '__all__', [])
for name in __all__:
    globals()[name] = getattr(_pkg, name)

# Expose submodules through this alias package
def _forward(name: str) -> None:
    module = import_module(f'ncps.ncps_mlx.{name}')
    sys.modules[f'{__name__}.{name}'] = module

def _forward_all():
    for name in ['wirings', 'ctgru_cell', 'ctgru', 'ctrnn_cell', 'ctrnn', 'eltc_cell', 'eltc', 'ltc4_cell', 'ltc4', 'datasets', 'liquid_utils', 'ltc_cell', 'ltc', 'ode_solvers', 'cfc_cell', 'cfc', 'wired_cfc_cell']:
        try:
            _forward(name)
        except ModuleNotFoundError:
            pass

_forward_all()
