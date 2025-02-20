Notebook Migration Guide
========================

This guide provides instructions for migrating existing notebooks to use
the latest MLX features and Apple Silicon optimizations.

Migration Overview
------------------

Goals
~~~~~

1. Improve performance on Apple Silicon
2. Ensure consistent notebook structure
3. Add hardware-specific optimizations
4. Enable proper performance monitoring

Benefits
~~~~~~~~

1. Better Neural Engine utilization
2. Improved memory efficiency
3. Consistent code organization
4. Hardware-specific optimizations

Step-by-Step Migration
----------------------

1. Update Notebook Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata Updates
^^^^^^^^^^^^^^^^

.. code:: json

{
    "metadata": {
    "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
    },
    "language_info": {
    "codemirror_mode": {
    "name": "ipython",
    "version": 3
    },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
"version": "3.8.0"
}}}}}}}}}}}}}}}}}}

Import Organization
^^^^^^^^^^^^^^^^^^^

.. code:: python

# Core imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

# NCP imports
from ncps.mlx import CfC, CfCCell
from ncps.wirings import AutoNCP
from ncps.mlx.advanced_profiling import MLXProfiler

# Device configuration
from ncps.tests.configs.device_configs import get_device_config

2. Add Device Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

# Get device configuration
config = get_device_config()
print(f"Detected device: {config.device_type}")
print(f"Optimal batch size: {config.get_optimal_batch_size()}")
print(f"Optimal hidden size: {config.get_optimal_hidden_size()}")
print(f"Optimal backbone: {config.get_optimal_backbone()}")

3. Update Model Creation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def create_optimized_model(config):
    """Create device-optimized model."""
    # Create wiring with optimal size
    wiring = AutoNCP(
        units=config.get_optimal_hidden_size(),
        output_size=config.get_optimal_hidden_size() // 4
    )

    # Create model with optimal backbone
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            activation="tanh",
            backbone_units=config.get_optimal_backbone(),
            backbone_layers=2
        ),
        return_sequences=True
    )

    return model

# Enable compilation
@mx.compile(static_argnums=(1,))
def forward(x, training=False):
    return model(x, training=training)

4. Add Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def profile_model(model, config):
    """Profile model performance."""
    profiler = MLXProfiler(model)

    # Profile compute
    compute_stats = profiler.profile_compute(
        batch_size=config.get_optimal_batch_size(),
        seq_length=16,
        num_runs=100
    )

    # Profile memory
    memory_stats = profiler.profile_memory(
        batch_size=config.get_optimal_batch_size()
    )

    return {
        'tflops': compute_stats['tflops'],
        'memory': memory_stats['peak_usage'],
        'bandwidth': memory_stats['bandwidth']
    }

5. Update Training Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def train_model(model, config, n_epochs=100):
    """Train with device-specific optimizations."""
    # Use optimal batch size
    batch_size = config.get_optimal_batch_size()

    # Create optimizer
    optimizer = optim.Adam(learning_rate=0.001)

    # Enable compilation
    @mx.compile(static_argnums=(1,))
    def train_step(x, training=True):
        def loss_fn(model, x, y):
            pred = model(x, training=training)
            return mx.mean((pred - y) ** 2)
        return mx.value_and_grad(model, loss_fn)

    # Training loop
    for epoch in range(n_epochs):
        # Training step implementation
        pass

Common Issues and Solutions
---------------------------

1. Performance Issues
~~~~~~~~~~~~~~~~~~~~~

- **Problem**: Low Neural Engine utilization
- **Solution**: Use power-of-2 sizes and enable compilation

2. Memory Issues
~~~~~~~~~~~~~~~~

- **Problem**: High memory usage
- **Solution**: Use device-specific batch sizes and monitor usage

3. Training Issues
~~~~~~~~~~~~~~~~~~

- **Problem**: Slow training speed
- **Solution**: Enable compilation and use optimal configurations

Testing Migration
-----------------

1. Notebook Validation
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def validate_notebook(model, config):
    """Validate notebook implementation."""
    # Check performance
    stats = profile_model(model, config)

    # Verify requirements
    assert stats['tflops'] >= config.min_tflops
    assert stats['bandwidth'] >= config.min_bandwidth
    assert stats['memory'] <= config.memory_budget

2. Performance Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

def verify_performance(model, config):
    """Verify model performance."""
    # Test with and without compilation
    uncompiled_stats = profile_uncompiled(model, config)
    compiled_stats = profile_compiled(model, config)

    # Verify speedup
    speedup = uncompiled_stats['time'] / compiled_stats['time']
    assert speedup >= 1.5  # Expect at least 1.5x speedup

Best Practices
--------------

1. **Code Organization**

- Use consistent import ordering
- Add clear section headers
- Include performance monitoring

2. **Performance Optimization**

- Use device-specific configurations
- Enable Neural Engine optimizations
- Monitor hardware utilization

3. **Documentation**

- Add hardware requirements
- Document optimization techniques
- Include performance tips

Migration Checklist
-------------------

- ☐ Update notebook metadata
- ☐ Organize imports
- ☐ Add device configuration
- ☐ Update model creation
- ☐ Add performance monitoring
- ☐ Update training functions
- ☐ Add validation tests
- ☐ Update documentation
- ☐ Test on different devices
- ☐ Verify performance

Resources
---------

1. MLX Documentation
2. Apple Silicon Developer Guide
3. Performance Optimization Guide
4. Hardware-Specific Examples
