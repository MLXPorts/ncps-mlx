"""Neural Circuit Policy implementations for MLX backend.

This module provides MLX implementations of various neural circuit components
including CfC (Closed-form Continuous-time) cells, LTC (Liquid Time-Constant) cells,
and their variants.
"""

from .cfc_cell_mlx import CfCCell
from .cfc import CfC
from .cfc_rnn import CfCRNN
from .cfc_rnn_cell import CfCRNNCell
from .ltc_cell import LTCCell
from .ltc import LTC
from .ltc_rnn import LTCRNN
from .ltc_rnn_cell import LTCRNNCell
from .eltc_cell import ELTCCell
from .mm_rnn import MMRNN
from .wired_cfc_cell import WiredCfCCell
from .wired_eltc_cell import WiredELTCCell
from .utils import save_model, load_model

__all__ = [
    # Base cells
    "CfCCell",
    "LTCCell",
    "ELTCCell",
    
    # Layer wrappers
    "CfC",
    "CfCRNN",
    "CfCRNNCell",
    "LTC",
    "LTCRNN",
    "LTCRNNCell",
    "MMRNN",
    
    # Wired variants
    "WiredCfCCell",
    "WiredELTCCell",
    
    # Utilities
    "save_model",
    "load_model",
]

# Version of the ncps.mlx package
__version__ = "1.0.0"
