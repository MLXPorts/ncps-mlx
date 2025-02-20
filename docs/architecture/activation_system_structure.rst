Activation System Structure
===========================

Overview
--------

Separate activation functions into individual files for better
organization and separation of concerns. Each activation type will have
its own file in both the base implementation and backend-specific
implementations.

Directory Structure
-------------------

::

ncps/
├── activations/
│   ├── __init__.py
│   ├── base.py           # Base activation class
│   ├── relu.py          # ReLU activation base
│   ├── sigmoid.py       # Sigmoid activation base
│   └── tanh.py          # Tanh activation base
└── mlx/
    └── activations/
        ├── __init__.py
        ├── relu.py      # MLX ReLU implementation
        ├── sigmoid.py   # MLX Sigmoid implementation
        └── tanh.py      # MLX Tanh implementation

Implementation Plan
-------------------

Base Classes
~~~~~~~~~~~~

base.py
^^^^^^^

.. code:: python

from abc import ABC, abstractmethod
from typing import Any

class Activation(ABC):
    @abstractmethod
    def __call__(self, x: Any) -> Any:
        pass

    @abstractmethod
    def gradient(self, x: Any) -> Any:
        pass

relu.py
^^^^^^^

.. code:: python

from .base import Activation

class ReLUBase(Activation):
    """Base class for ReLU activation."""
    pass

Similar structure for sigmoid.py and tanh.py.

MLX Implementation
~~~~~~~~~~~~~~~~~~

.. _relu.py-1:

relu.py
^^^^^^^

.. code:: python

import mlx.core as mx
from ncps.activations.relu import ReLUBase

class ReLU(ReLUBase):
    def __call__(self, x):
        return mx.maximum(0, x)

    def gradient(self, x):
        return mx.where(x > 0, 1.0, 0.0)

Similar structure for sigmoid.py and tanh.py.

Benefits
--------

1. Clear separation of concerns
2. Each activation function is self-contained
3. Easier to maintain and extend
4. Better organization of code
5. Simpler to add new activation functions
6. Clearer inheritance structure

Usage Example
-------------

\```python from ncps.mlx.activations import ReLU

activation = ReLU() result = activation(input_tensor) gradient =
activation.gradient(input_tensor)
