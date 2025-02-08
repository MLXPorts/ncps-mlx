from ncps.mini_keras.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.backend.common.dtypes import result_type
from ncps.mini_keras.backend.common.keras_tensor import KerasTensor
from ncps.mini_keras.backend.common.keras_tensor import any_symbolic_tensors
from ncps.mini_keras.backend.common.keras_tensor import is_keras_tensor
from ncps.mini_keras.backend.common.masking import get_keras_mask
from ncps.mini_keras.backend.common.masking import set_keras_mask
from ncps.mini_keras.backend.common.stateless_scope import StatelessScope
from ncps.mini_keras.backend.common.stateless_scope import get_stateless_scope
from ncps.mini_keras.backend.common.stateless_scope import in_stateless_scope
from ncps.mini_keras.backend.common.symbolic_scope import SymbolicScope
from ncps.mini_keras.backend.common.symbolic_scope import in_symbolic_scope
from ncps.mini_keras.backend.common.variables import AutocastScope
from ncps.mini_keras.backend.common.variables import Variable
from ncps.mini_keras.backend.common.variables import get_autocast_scope
from ncps.mini_keras.backend.common.variables import is_float_dtype
from ncps.mini_keras.backend.common.variables import is_int_dtype
from ncps.mini_keras.backend.common.variables import standardize_dtype
from ncps.mini_keras.backend.common.variables import standardize_shape
from ncps.mini_keras.backend.common.numpy import pi
from ncps.mini_keras.backend.config import epsilon
from ncps.mini_keras.backend.config import floatx
from ncps.mini_keras.backend.config import image_data_format
from ncps.mini_keras.backend.config import set_epsilon
from ncps.mini_keras.backend.config import set_floatx
from ncps.mini_keras.backend.config import set_image_data_format
from ncps.mini_keras.backend.config import standardize_data_format

# Import backend functions.
if backend() == "tensorflow":
    from ncps.mini_keras.backend.tensorflow import *  # noqa: F403
    from ncps.mini_keras.backend.tensorflow.core import Variable as BackendVariable
elif backend() == "jax":
    from ncps.mini_keras.backend.jax import *  # noqa: F403
    from ncps.mini_keras.backend.jax.core import Variable as BackendVariable
elif backend() == "torch":
    from ncps.mini_keras.backend.torch import *  # noqa: F403
    from ncps.mini_keras.backend.torch.core import Variable as BackendVariable
    distribution_lib = None
elif backend() == "numpy":
    from ncps.mini_keras.backend.numpy import *  # noqa: F403
    from ncps.mini_keras.backend.numpy.core import Variable as BackendVariable
    distribution_lib = None
elif backend() == "openvino":
    from ncps.mini_keras.backend.openvino import *  # noqa: F403
    from ncps.mini_keras.backend.openvino.core import Variable as BackendVariable
    distribution_lib = None
elif backend() == "mlx":
    from ncps.mini_keras.backend.mlx import * # noqa: F403
    from ncps.mini_keras.backend.mlx.core import Variable as BackendVariable
    distribution_lib = None
else:
    raise ValueError(f"Unable to import backend : {backend()}")

__all__ = [ "keras_mini_export", "result_type", "KerasTensor", "any_symbolic_tensors", 
           "is_keras_tensor", "get_keras_mask", "set_keras_mask", "StatelessScope", "get_stateless_scope", 
           "in_stateless_scope", "SymbolicScope", "in_symbolic_scope", "AutocastScope", "Variable", 
           "get_autocast_scope", "is_float_dtype", "is_int_dtype", "standardize_dtype", "standardize_shape", 
           "epsilon", "floatx", "image_data_format", "set_epsilon", "set_floatx", "set_image_data_format", 
           "standardize_data_format", "distribution_lib", "Variable", "name_scope", "device", 
           "backend_name_scope", "torch", "pi"]

@keras_mini_export("ncps.keras_mini.Variable")
class Variable(BackendVariable):  # noqa: F811
    pass


backend_name_scope = name_scope  # noqa: F405


@keras_mini_export("ncps.keras_mini.name_scope")
class name_scope(backend_name_scope):
    pass


@keras_mini_export("ncps.keras_mini.device")
def device(device_name):
    return device_scope(device_name)  # noqa: F405
