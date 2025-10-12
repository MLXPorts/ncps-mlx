:hide-toc:

===================================================
Welcome to Neural Circuit Policies for Apple MLX!
===================================================

`Neural Circuit Policies (NCPs) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ are machine learning models inspired by the nervous system of the nematode *C. elegans*.
This package provides an easy-to-use MLX implementation of NCPs, specifically optimized for Apple Silicon (M1/M2/M3/M4 chips).

This is a **complete MLX rewrite** of the original PyTorch/TensorFlow implementation, providing native GPU acceleration on Apple Silicon through the MLX framework.

.. code-block:: bash

    pip3 install ncps-mlx

MLX Example:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC
    from ncps.wirings import AutoNCP

    # Create a CfC model with 20 input features and 50 hidden units
    rnn = CfC(input_size=20, units=50)

    # Or use structured NCP wiring
    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
    rnn = CfC(input_size=20, units=wiring)

    # Forward pass
    x = mx.random.normal((2, 3, 20))  # (batch, time, features)
    output, state = rnn(x)

Key Features
--------------

- **🚀 Optimized for Apple Silicon**: Native MLX implementation for M1/M2/M3/M4 chips
- **🧠 Multiple RNN Architectures**: LTC, CfC, CTRNN, CTGRU
- **🔌 Flexible Wiring**: Support for structured NCP wirings and fully-connected layers
- **📊 Production Ready**: Includes model checkpointing, state management, and profiling
- **🎯 Type Safe**: Full type annotations for better development experience
- **📈 Easy Training**: Compatible with standard MLX training patterns

About the MLX Port
--------------------

This implementation differs from the original PyTorch/TensorFlow version:

**Architecture Changes:**

- **MLX-Native Operations**: All operations use MLX primitives for optimal Apple Silicon performance
- **Unified API**: Single, consistent API (no separate torch/tf modules)
- **Direct Import**: ``from ncps import LTC, CfC`` (no need for ``ncps.mlx``)
- **Batch-First by Default**: Follows MLX conventions with ``(batch, time, features)`` ordering

**Performance Optimizations:**

- **Apple Neural Engine**: Automatically leverages Apple's specialized ML hardware
- **Unified Memory**: Takes advantage of Apple Silicon's unified memory architecture
- **Metal Backend**: Uses Metal for GPU acceleration on M-series chips

User's Guide
--------------

.. toctree::
    :maxdepth: 2

    quickstart
    examples/index
    api/index

Attribution
-----------

This project is based on the original `ncps <https://github.com/mlech26l/ncps>`_ library by Mathias Lechner and contributors. 
The MLX version is a ground-up rewrite by Sydney Renee for The Solace Project, specifically optimized for Apple Silicon.

**Original Research Papers:**

- `Neural Circuit Policies Enabling Auditable Autonomy <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_
- `Closed-form continuous-time neural networks <https://www.nature.com/articles/s42256-022-00556-7>`_
