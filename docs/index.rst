Welcome to NCPS Documentation
=============================

Neural Circuit Policies (NCPS) is a framework for implementing and training liquid neural networks, with support for multiple backends including MLX, PyTorch, and TensorFlow.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/abstractions/index
   architecture/implementation/index
   architecture/knowledge/index
   architecture/design/index
   architecture/research/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/mlx
   api/torch
   api/keras
   api/wirings

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/performance_optimization
   guides/advanced_features
   guides/visualization
   guides/deployment

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   roadmap

Features
--------

- Multiple backend support (MLX, PyTorch, TensorFlow)
- Liquid neural network implementations
- Hardware acceleration abstraction
- Flexible layer technology system
- Comprehensive visualization tools

Quick Example
-------------

.. code-block:: python

    from ncps.mlx import CfC
    import mlx.core as mx

    # Create a model
    model = CfC(
        units=64,
        return_sequences=True
    )

    # Generate sample data
    x = mx.random.normal((32, 100, 10))  # (batch_size, time_steps, features)
    
    # Process data
    y = model(x)
    print(f"Output shape: {y.shape}")

Installation
------------

.. code-block:: bash

    pip install ncps-mlx

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
