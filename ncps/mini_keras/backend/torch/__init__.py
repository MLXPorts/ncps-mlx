"""Torch backend APIs.

# Note on device placement

Torch has a different device placement style compared to TF and JAX.
In short, variables/tensors are not created on GPU by default,
and the GPU cannot directly communicate with the CPU.
To bring Torch behavior in line with TF and JAX automated device placement,
we are doing the following to automate device placement if a GPU is available:

- Variables are created on GPU.
- Input data will be placed on GPU at the first `keras.layers.Layer` call.
- Tensor creation happens on GPU, e.g., `zeros()` will create a tensor on GPU.
- `convert_to_numpy` will bring the tensor to CPU before converting it to NumPy.
"""

from ncps.mini_keras.backend.common.name_scope import name_scope
from ncps.mini_keras.backend.torch import core
from ncps.mini_keras.backend.torch import image
from ncps.mini_keras.backend.torch import linalg
from ncps.mini_keras.backend.torch import math
from ncps.mini_keras.backend.torch import nn
from ncps.mini_keras.backend.torch import numpy
from ncps.mini_keras.backend.torch import random
from ncps.mini_keras.backend.torch.core import IS_THREAD_SAFE
from ncps.mini_keras.backend.torch.core import SUPPORTS_RAGGED_TENSORS
from ncps.mini_keras.backend.torch.core import SUPPORTS_SPARSE_TENSORS
from ncps.mini_keras.backend.torch.core import Variable
from ncps.mini_keras.backend.torch.core import cast
from ncps.mini_keras.backend.torch.core import compute_output_spec
from ncps.mini_keras.backend.torch.core import cond
from ncps.mini_keras.backend.torch.core import convert_to_numpy
from ncps.mini_keras.backend.torch.core import convert_to_tensor
from ncps.mini_keras.backend.torch.core import device_scope
from ncps.mini_keras.backend.torch.core import is_tensor
from ncps.mini_keras.backend.torch.core import random_seed_dtype
from ncps.mini_keras.backend.torch.core import scatter
from ncps.mini_keras.backend.torch.core import shape
from ncps.mini_keras.backend.torch.core import stop_gradient
from ncps.mini_keras.backend.torch.core import to_torch_dtype
from ncps.mini_keras.backend.torch.core import vectorized_map
from ncps.mini_keras.backend.torch.rnn import cudnn_ok
from ncps.mini_keras.backend.torch.rnn import gru
from ncps.mini_keras.backend.torch.rnn import lstm
from ncps.mini_keras.backend.torch.rnn import rnn

__all__ = [ "core", "image", "linalg", "math", "nn", "numpy", "random", "IS_THREAD_SAFE", "SUPPORTS_RAGGED_TENSORS", "SUPPORTS_SPARSE_TENSORS", "Variable", "cast", "compute_output_spec", "cond", "convert_to_numpy", "convert_to_tensor", "device_scope", "is_tensor", "name_scope", "random_seed_dtype", "scatter", "shape", "stop_gradient", "to_torch_dtype", "vectorized_map", "cudnn_ok", "gru", "lstm", "rnn", ]

