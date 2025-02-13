"""Neural Circuit Policies (NCPs) Keras implementation."""

import keras
from keras import activations

# Register custom activation function
@keras.saving.register_keras_serializable(package="ncps")
def lecun_tanh(x):
    """LeCun improved tanh activation."""
    return 1.7159 * activations.tanh(0.666 * x)

# Add to Keras activations
keras.utils.get_custom_objects()['lecun_tanh'] = lecun_tanh
activations.lecun_tanh = lecun_tanh

# Import base classes
from .base import LiquidCell, LiquidRNN

# Import wiring patterns
from .wirings import (
    Wiring,
    FullyConnected,
    Random,
    NCP,
    AutoNCP,
)

# Import cell implementations
from .cfc_cell import CfCCell
from .ltc_cell import LTCCell

# Import RNN implementations
from .cfc import CfC
from .ltc import LTC

# Register custom objects
keras.saving.register_keras_serializable(package="ncps")(LiquidCell)
keras.saving.register_keras_serializable(package="ncps")(LiquidRNN)
keras.saving.register_keras_serializable(package="ncps")(Wiring)
keras.saving.register_keras_serializable(package="ncps")(FullyConnected)
keras.saving.register_keras_serializable(package="ncps")(Random)
keras.saving.register_keras_serializable(package="ncps")(NCP)
keras.saving.register_keras_serializable(package="ncps")(AutoNCP)
keras.saving.register_keras_serializable(package="ncps")(CfCCell)
keras.saving.register_keras_serializable(package="ncps")(LTCCell)
keras.saving.register_keras_serializable(package="ncps")(CfC)
keras.saving.register_keras_serializable(package="ncps")(LTC)

# Version
__version__ = "2.0.0"

__all__ = [
    # Base classes
    "LiquidCell",
    "LiquidRNN",
    
    # Wiring patterns
    "Wiring",
    "FullyConnected",
    "Random",
    "NCP",
    "AutoNCP",
    
    # Cell implementations
    "CfCCell",
    "LTCCell",
    
    # RNN implementations
    "CfC",
    "LTC",
    
    # Utilities
    "lecun_tanh",
]
