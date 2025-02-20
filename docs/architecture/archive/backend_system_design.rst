Backend System Design
=====================

Current Approach Issues
-----------------------

The current approach of using MLX-prefixed classes has several
drawbacks: 1. Verbose naming convention 2. Requires explicit imports
from specific backend packages 3. Makes switching backends at runtime
difficult 4. Code becomes tightly coupled to specific backends

Alternative Approaches
----------------------

1. Class Registration System (Keras-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class BackendRegistry:
    _backends = {}
    _active_backend = None

    @classmethod
    def register(cls, backend_name, layer_type, implementation):
        if backend_name not in cls._backends:
            cls._backends[backend_name] = {}
        cls._backends[backend_name][layer_type] = implementation

    @classmethod
    def get(cls, layer_type):
        return cls._backends[cls._active_backend][layer_type]

# Usage
@register_backend("mlx", "Dense")
class MLXDense(DenseBase):
    pass

# Get implementation
Dense = BackendRegistry.get("Dense")

Pros: - Clean API - Runtime backend switching - No backend-specific
imports needed - Consistent naming

Cons: - More complex implementation - Potential for registration
conflicts - Need to manage backend state

2. Factory Pattern with Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class LayerFactory:
    @staticmethod
    def create(layer_type, backend=None, **kwargs):
        backend = backend or get_default_backend()
        implementation = get_implementation(backend, layer_type)
        return implementation(**kwargs)

# Usage
layer = LayerFactory.create("Dense", units=64)

Pros: - Explicit creation - Flexible configuration - Easy to extend

Cons: - Less intuitive API - Still needs backend management - Factory
could become complex

3. Dynamic Import System
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def get_layer(name, backend=None):
    backend = backend or get_default_backend()
    module = importlib.import_module(f"ncps.{backend}.layers")
    return getattr(module, name)

# Usage
Dense = get_layer("Dense")
layer = Dense(units=64)

Pros: - Simple implementation - Flexible - Clear separation

Cons: - Import overhead - Less IDE support - Potential import errors

4. Context-Based System
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class BackendContext:
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        set_active_backend(self.backend)

    def __exit__(self, *args):
        restore_previous_backend()

# Usage
with BackendContext("mlx"):
    layer = Dense(units=64)

Pros: - Clear scope - Easy to switch - Clean API

Cons: - Context management overhead - Potential for confusion - State
management complexity

Recommended Approach: Hybrid Registration System
------------------------------------------------

Combine the best aspects of the registration system and context manager:

.. code:: python

class NCPSBackend:
    _registry = {}
    _active_backend = None

    @classmethod
    def register_implementation(cls, backend, layer_type, implementation):
        if backend not in cls._registry:
            cls._registry[backend] = {}
        cls._registry[backend][layer_type] = implementation

    @classmethod
    def set_backend(cls, backend):
        if backend not in cls._registry:
            raise ValueError(f"Backend {backend} not registered")
        cls._active_backend = backend

    @classmethod
    def get_implementation(cls, layer_type):
        if cls._active_backend is None:
            raise RuntimeError("No active backend set")
        return cls._registry[cls._active_backend][layer_type]

# Usage
class Dense(LayerBase):
    def __new__(cls, *args, **kwargs):
        impl = NCPSBackend.get_implementation("Dense")
        return impl(*args, **kwargs)

# Registration
NCPSBackend.register_implementation("mlx", "Dense", MLXDenseImpl)
NCPSBackend.set_backend("mlx")

# Use anywhere
layer = Dense(units=64)  # Gets MLX implementation

Benefits
~~~~~~~~

1. Clean API without backend prefixes
2. Runtime backend switching
3. Automatic implementation selection
4. Type hints and IDE support work
5. No need for explicit imports from backend packages
6. Easy to add new backends
7. Clear separation of concerns

Implementation Steps
~~~~~~~~~~~~~~~~~~~~

1. Create NCPSBackend system
2. Move implementations to backend-specific packages
3. Register implementations
4. Update base classes to use **new** for implementation selection
5. Add backend configuration system
6. Update documentation and examples

This approach provides the flexibility of runtime backend switching
while maintaining a clean API and good developer experience.
