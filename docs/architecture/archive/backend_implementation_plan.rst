Backend System Implementation Plan
==================================

Phase 1: Core Backend System
----------------------------

1. Create Backend Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new module ``ncps/backend.py``:

.. code:: python

from typing import Dict, Type, Any, Optional
from contextlib import contextmanager

class NCPSBackend:
    """Central backend management system for NCPS."""

    _registry: Dict[str, Dict[str, Type[Any]]] = {}
    _active_backend: Optional[str] = None
    _backend_stack: List[str] = []

    @classmethod
    def register(cls, backend: str, layer_type: str, implementation: Type[Any]) -> None:
        """Register a backend implementation for a layer type."""
        if backend not in cls._registry:
            cls._registry[backend] = {}
        cls._registry[backend][layer_type] = implementation

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """Set the active backend."""
        if backend not in cls._registry:
            raise ValueError(f"Backend {backend} not registered")
        cls._active_backend = backend

    @classmethod
    def get_backend(cls) -> str:
        """Get the current active backend."""
        if cls._active_backend is None:
            raise RuntimeError("No active backend set")
        return cls._active_backend

    @classmethod
    def get_implementation(cls, layer_type: str) -> Type[Any]:
        """Get the implementation for a layer type in the active backend."""
        backend = cls.get_backend()
        if layer_type not in cls._registry[backend]:
            raise ValueError(f"No implementation for {layer_type} in backend {backend}")
        return cls._registry[backend][layer_type]

    @classmethod
    @contextmanager
    def backend_scope(cls, backend: str):
        """Temporarily switch backend within a context."""
        cls._backend_stack.append(cls._active_backend)
        cls.set_backend(backend)
        try:
            yield
        finally:
            cls._active_backend = cls._backend_stack.pop()

2. Create Base Layer System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update base layer classes to use the backend system:

.. code:: python

# ncps/layers/base.py
from abc import ABC, abstractmethod
from ncps.backend import NCPSBackend

class Layer(ABC):
    def __new__(cls, *args, **kwargs):
        if cls == Layer:
            raise TypeError("Layer class cannot be instantiated directly")
        if cls.__name__ in NCPSBackend._registry[NCPSBackend.get_backend()]:
            impl = NCPSBackend.get_implementation(cls.__name__)
            return impl(*args, **kwargs)
        return super().__new__(cls)

Phase 2: Backend Implementations
--------------------------------

1. MLX Backend Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create MLX implementations and register them:

.. code:: python

# ncps/mlx/layers.py
from ncps.backend import NCPSBackend
from ncps.layers.base import Layer

class MLXDense(Layer):
    def __init__(self, units, **kwargs):
        # MLX-specific implementation
        pass

# Register implementation
NCPSBackend.register("mlx", "Dense", MLXDense)

2. Default Backend Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up default backend selection:

.. code:: python

# ncps/__init__.py
from ncps.backend import NCPSBackend

def _initialize_backend():
    """Initialize default backend based on environment."""
    try:
        import mlx.core
        NCPSBackend.set_backend("mlx")
    except ImportError:
        try:
            import torch
            NCPSBackend.set_backend("torch")
        except ImportError:
            try:
                import tensorflow
                NCPSBackend.set_backend("tensorflow")
            except ImportError:
                raise RuntimeError("No supported backend found")

_initialize_backend()

Phase 3: Layer Implementation
-----------------------------

1. Base Layer Types
~~~~~~~~~~~~~~~~~~~

Define layer interfaces without implementation:

.. code:: python

# ncps/layers/__init__.py
from ncps.backend import NCPSBackend

class Dense:
    def __new__(cls, *args, **kwargs):
        impl = NCPSBackend.get_implementation("Dense")
        return impl(*args, **kwargs)

2. Backend-Specific Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organize backend implementations:

.. code:: python

# ncps/mlx/layers/dense.py
import mlx.core as mx
from ncps.layers.base import Layer

class MLXDense(Layer):
    # MLX-specific implementation
    pass

# Register with backend
NCPSBackend.register("mlx", "Dense", MLXDense)

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code:: python

from ncps.layers import Dense

# Uses active backend automatically
layer = Dense(units=64)

Temporary Backend Switch
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

from ncps.backend import NCPSBackend

# Temporarily use MLX backend
with NCPSBackend.backend_scope("mlx"):
    layer = Dense(units=64)

Global Backend Switch
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

from ncps.backend import NCPSBackend

# Switch to MLX backend globally
NCPSBackend.set_backend("mlx")

Migration Steps
---------------

1. Implement Core System

- Create backend.py
- Update base layer classes
- Set up backend initialization

2. Update Existing Layers

- Move implementations to backend packages
- Register implementations with backend system
- Update layer interfaces

3. Documentation & Testing

- Update documentation with new usage patterns
- Add tests for backend switching
- Add examples using different backends

4. Cleanup

- Remove old backend-specific prefixes
- Update import statements
- Remove redundant code

This implementation provides a clean, flexible system for managing
multiple backends while maintaining a simple API for users.
