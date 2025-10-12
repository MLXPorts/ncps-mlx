===================================================
MLX Quick Start
===================================================

Getting Started with NCPs in MLX
=================================

This guide demonstrates the basic usage of Neural Circuit Policies with Apple's MLX framework.

Installation
------------

First, ensure you have MLX and ncps-mlx installed:

.. code-block:: bash

    pip install mlx
    pip install ncps-mlx

Basic Example: Fully-Connected CfC
===================================

Here's a simple example using a fully-connected CfC model:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC

    # Create a fully-connected CfC model
    input_size = 20
    hidden_units = 50
    rnn = CfC(input_size=input_size, units=hidden_units)

    # Create some random input data
    # Shape: (batch_size, sequence_length, input_size)
    batch_size = 2
    seq_length = 10
    x = mx.random.normal((batch_size, seq_length, input_size))

    # Forward pass
    output, hidden_state = rnn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden_state.shape}")

Using Structured NCP Wirings
=============================

Neural Circuit Policies shine when using structured wirings:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC
    from ncps.wirings import AutoNCP

    # Create an AutoNCP wiring with 28 neurons and 4 outputs
    wiring = AutoNCP(units=28, output_size=4)
    
    # Create a CfC model with this wiring
    input_size = 20
    rnn = CfC(input_size=input_size, units=wiring)

    # Forward pass
    x = mx.random.normal((2, 10, input_size))
    output, hidden_state = rnn(x)

    print(f"Output shape: {output.shape}")  # (2, 10, 4)

Using LTC Models
================

The Liquid Time-Constant (LTC) model is also available:

.. code-block:: python

    import mlx.core as mx
    from ncps import LTC
    from ncps.wirings import AutoNCP

    # Create an LTC model with NCP wiring
    wiring = AutoNCP(units=32, output_size=8)
    rnn = LTC(input_size=20, units=wiring)

    # Forward pass
    x = mx.random.normal((4, 15, 20))
    output, hidden_state = rnn(x)

Controlling Sequence Output
============================

By default, models return the full sequence. You can return only the last output:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC

    # Return only the last timestep output
    rnn = CfC(input_size=20, units=50, return_sequences=False)

    x = mx.random.normal((2, 10, 20))
    output, hidden_state = rnn(x)

    print(f"Output shape: {output.shape}")  # (2, 50) - only last timestep

Building a Classifier
======================

Here's a complete example of a sequence classifier:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    from ncps import CfC
    from ncps.wirings import AutoNCP

    class SequenceClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            # Use AutoNCP wiring for structured connectivity
            wiring = AutoNCP(hidden_dim, hidden_dim)
            self.rnn = CfC(
                input_size=input_dim, 
                units=wiring, 
                return_sequences=False  # Only last timestep
            )
            self.classifier = nn.Linear(self.rnn.output_size, num_classes)
        
        def __call__(self, x):
            # Get RNN output for last timestep
            rnn_out, _ = self.rnn(x)
            # Classify
            return self.classifier(rnn_out)

    # Create model
    model = SequenceClassifier(
        input_dim=20, 
        hidden_dim=64, 
        num_classes=10
    )

    # Test forward pass
    x = mx.random.normal((32, 100, 20))  # (batch, time, features)
    logits = model(x)
    print(f"Logits shape: {logits.shape}")  # (32, 10)

Next Steps
==========

- See :doc:`mlx_training` for training examples
- Check the `examples/ directory <https://github.com/SolaceHarmony/ncps-mlx/tree/master/examples>`_ for more complete examples
- Read the :doc:`../quickstart` guide for more details on wirings and models
