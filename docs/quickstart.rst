:hide-toc:

===================================================
Quickstart
===================================================

Neural Circuit Policies are recurrent neural network models inspired by the nervous system of the nematode C. elegans.
Compared to standard ML models, NCPs have:

#. Neurons that are modeled by an ordinary differential equation
#. A sparse structured wiring

This MLX implementation provides native GPU acceleration on Apple Silicon (M1/M2/M3/M4 chips) through Apple's MLX framework.

Neuron Models
=============================
The package currently provides two primary neuron models: **LTC** and **CfC**:

The `liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936/16743>`_ model is based on neurons in the form of differential equations interconnected via sigmoidal synapses.
The term liquid time-constant comes from the property of LTCs that their timing behavior is adaptive to the input (how fast or slow they respond to some stimulus can depend on the specific input).
Because LTCs are ordinary differential equations, their behavior can only be described over time.
LTCs are universal approximators and implement causal dynamical models.
However, the LTC model has one major disadvantage: to compute their output, we need a numerical differential equation solver which seriously slows down their training and inference time.

**Closed-form continuous-time (CfC)** models resolve this bottleneck by approximating the closed-form solution of the differential equation.

.. note::
    Both the LTC and the CfC models are **recurrent neural networks** and possess a temporal state. Therefore, these models are applicable only to sequential or time-series data.

Additional Models
=============================
The package also includes:

- **CTRNN** - Continuous-time recurrent neural networks
- **CTGRU** - Continuous-time gated recurrent units

Basic Usage
=============================

Fully-Connected Models
-----------------------

We can use both LTC and CfC models with a fully-connected wiring diagram by simply passing the number of neurons, i.e., as it is done in standard ML models such as LSTMs, GRU, MLPs, and Transformers.

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC, LTC

    # A fully connected CfC network
    rnn = CfC(input_size=20, units=50)
    
    # A fully connected LTC network
    rnn = LTC(input_size=20, units=50)

    # Forward pass
    x = mx.random.normal((2, 3, 20))  # (batch, time, features)
    output, state = rnn(x)

Structured NCP Wirings
-----------------------

We can also specify sparse structured wirings in the form of a ``ncps.wirings.Wiring`` object.
The `Neural Circuit Policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ is the most interesting wiring paradigm provided in this package and comprises of a 4-layer recurrent connection principle of sensory, inter, command, and motor neurons.

.. image:: ./img/wirings.png
   :align: center

The easiest way to create a NCP wiring is via the ``AutoNCP`` class, which requires defining the total number of neurons and the number of motor neurons (= output size).

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
    input_size = 20
    rnn = CfC(input_size, wiring)

    # Forward pass
    x = mx.random.normal((2, 3, 20))  # (batch, time, features)
    output, state = rnn(x)

Training Example
=============================

Here's a complete example of training a CfC model with MLX:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from ncps import CfC
    from ncps.wirings import AutoNCP

    # Create model with structured NCP wiring
    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
    model = CfC(input_size=20, units=wiring, return_sequences=True)

    # Define loss function
    def loss_fn(model, x, y):
        outputs, _ = model(x)
        return mx.mean((outputs - y) ** 2)

    # Training loop
    optimizer = optim.Adam(learning_rate=1e-3)
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(100):
        # Get loss and gradients
        loss, grads = value_and_grad_fn(model, x_train, y_train)
        
        # Update model parameters
        optimizer.update(model, grads)
        
        # Evaluate to ensure updates are applied
        mx.eval(model.parameters(), optimizer.state)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Building Sequential Models
============================

You can combine NCP models with other MLX layers to build more complex architectures:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    from ncps import CfC
    from ncps.wirings import AutoNCP

    class SequenceClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            wiring = AutoNCP(hidden_dim, hidden_dim)
            self.rnn = CfC(input_size=input_dim, units=wiring, return_sequences=False)
            self.output = nn.Linear(self.rnn.output_size, num_classes)
        
        def __call__(self, x):
            features, _ = self.rnn(x)
            return self.output(features)

    # Create and use model
    model = SequenceClassifier(input_dim=20, hidden_dim=64, num_classes=10)
    x = mx.random.normal((32, 100, 20))  # (batch, time, features)
    logits = model(x)

Key Differences from PyTorch/TensorFlow Version
=================================================

This MLX implementation has several key differences:

**API Changes:**

- Uses ``mlx.core`` arrays instead of PyTorch tensors or TensorFlow tensors
- Direct import: ``from ncps import CfC, LTC`` (no need for ``ncps.mlx`` or ``ncps.torch``)
- Batch-first by default: ``(batch, time, features)`` ordering
- Returns tuple ``(output, state)`` from forward pass

**Performance:**

- Optimized for Apple Silicon with Metal backend
- Leverages unified memory architecture
- Automatic Apple Neural Engine utilization

**State Management:**

- Cleaner hidden state handling
- Native support for MLX weight serialization
- Simplified model checkpointing

Diagram
=============================

.. image:: ./img/things.png
   :align: center

The diagram above shows the structured wiring of a Neural Circuit Policy, with sensory neurons (blue) receiving inputs, inter neurons (green) processing information, command neurons (orange) making decisions, and motor neurons (red) producing outputs.
