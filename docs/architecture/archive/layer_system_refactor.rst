Layer System Refactor Plan
==========================

Overview
--------

Refactor the layer system to use a similar structure as the activation
system, with base classes in the core package and backend-specific
implementations in their respective packages.

Current Structure Analysis
--------------------------

Currently we have: 1. Base layer functionality in layer.py (Layer,
Dense, Sequential) 2. Liquid neural network base in base.py
(BackboneLayerCell) 3. Specific implementations (CFC, CTGRU, CTRNN,
etc.) in separate files

Proposed Directory Structure
----------------------------

::

ncps/
├── layers/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── layer.py         # Core Layer class
│   │   ├── dense.py         # Base Dense layer
│   │   ├── sequential.py    # Base Sequential container
│   │   └── backbone.py      # Base BackboneLayerCell
│   ├── liquid/
│   │   ├── __init__.py
│   │   ├── cfc.py          # Base CFC layer
│   │   ├── ctgru.py        # Base CTGRU layer
│   │   ├── ctrnn.py        # Base CTRNN layer
│   │   ├── eltc.py         # Base ELTC layer
│   │   └── ltc.py          # Base LTC layer
│   └── utils/
│       ├── __init__.py
│       ├── liquid_utils.py
│       └── ode_solvers.py
└── mlx/
    └── layers/
        ├── __init__.py
        ├── base/
        │   ├── __init__.py
        │   ├── layer.py     # MLX Layer implementation
        │   ├── dense.py     # MLX Dense implementation
        │   └── sequential.py # MLX Sequential implementation
        └── liquid/
            ├── __init__.py
            ├── cfc.py       # MLX CFC implementation
            ├── ctgru.py     # MLX CTGRU implementation
            ├── ctrnn.py     # MLX CTRNN implementation
            ├── eltc.py      # MLX ELTC implementation
            └── ltc.py       # MLX LTC implementation

Implementation Plan
-------------------

1. Base Layer Classes
~~~~~~~~~~~~~~~~~~~~~

Layer Base Class (layers/base/layer.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List

class Layer(ABC):
    """Abstract base class for all layers."""

    @abstractmethod
    def __call__(self, inputs: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def build(self, input_shape: Any) -> None:
        pass

Dense Base Class (layers/base/dense.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

from .layer import Layer

class DenseBase(Layer):
    """Base class for dense layers."""

    def __init__(self, units: int, activation: Optional[str] = None):
        self.units = units
        self.activation = activation

2. MLX Implementations
~~~~~~~~~~~~~~~~~~~~~~

MLX Layer (mlx/layers/base/layer.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

from ncps.layers.base.layer import Layer
import mlx.core as mx

class MLXLayer(Layer):
    """MLX implementation of base layer."""

    def build(self, input_shape):
        # MLX-specific implementation
        pass

MLX Dense (mlx/layers/base/dense.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

from ncps.layers.base.dense import DenseBase
import mlx.core as mx

class Dense(DenseBase):
    """MLX implementation of dense layer."""

    def __call__(self, inputs):
        # MLX-specific implementation
        pass

Benefits
--------

1. Clear separation between base interfaces and implementations
2. Each layer type is self-contained
3. Better organization of liquid neural network components
4. Easier to maintain and extend
5. Simpler to add new backend implementations
6. Clearer inheritance structure
7. Utils properly separated

Migration Steps
---------------

1. Create new directory structure
2. Move base classes to appropriate locations
3. Create MLX implementations
4. Update imports throughout codebase
5. Add proper type hints and documentation
6. Update tests to reflect new structure

Usage Example
-------------

\```python # Base interface from ncps.layers.base import DenseBase

MLX implementation
==================

from mlx import nn

Create and use layer
====================

layer = nn.Linear(in_features, 64)
output = nn.relu(layer(input_tensor))
