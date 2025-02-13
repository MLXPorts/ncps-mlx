:hide-toc:

===================================================
Welcome to Neural Circuit Policies's documentation!
===================================================

`Neural Circuit Policies (NCPs) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ are machine learning models inspired by the nervous system of the nematode *C. elegans*.
This package provides optimized implementations of NCPs for MLX (Apple Silicon), PyTorch, and TensorFlow.

.. code-block:: bash

    pip3 install -U ncps

MLX Example (Apple Silicon):

.. code-block:: python

    from ncps.mlx import CfC, CfCCell
    from ncps.wirings import AutoNCP
    import mlx.core as mx

    # Create wiring
    wiring = AutoNCP(units=32, output_size=4)

    # Create model optimized for Apple Silicon
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            activation="tanh",
            backbone_units=[64, 64],  # Power of 2 sizes
            backbone_layers=2
        ),
        return_sequences=True
    )

    # Process sequence
    x = mx.random.normal((32, 10, 8))  # (batch, time, features)
    outputs = model(x)

PyTorch Example:

.. code-block:: python

    from ncps.torch import CfC

    # A fully connected CfC network
    rnn = CfC(input_size=20, units=50)
    x = torch.randn(2, 3, 20)  # (batch, time, features)
    h0 = torch.zeros(2, 50)  # (batch, units)
    output, hn = rnn(x, h0)

TensorFlow Example:

.. code-block:: python

    from ncps.tf import LTC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 2)),
        LTC(wiring, return_sequences=True)
    ])

Installation Guide
-------------------

To install the `ncps` package:

1. Ensure you have Python 3.8 or higher installed.
2. Install the base package:

.. code-block:: bash

    pip install ncps

3. Install framework-specific dependencies:

For MLX (Apple Silicon):

.. code-block:: bash

    pip install mlx

For PyTorch:

.. code-block:: bash

    pip install torch

For TensorFlow:

.. code-block:: bash

    pip install tensorflow

4. Optional dependencies for documentation and examples:

.. code-block:: bash

    pip install sphinx numpy sphinx-gallery numpydoc pandas sphinx-copybutton furo sphinx_design

Examples and Use Cases
----------------------

1. Apple Silicon Optimization:

.. code-block:: python

    from ncps.mlx import CfC, CfCCell
    from ncps.wirings import AutoNCP

    # Create device-optimized model
    wiring = AutoNCP(
        units=128,  # Power of 2
        output_size=32
    )

    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            backbone_units=[128, 128],  # Power of 2
            backbone_layers=2
        )
    )

    # Enable compilation for Neural Engine
    @mx.compile(static_argnums=(1,))
    def forward(x, training=False):
        return model(x, training=training)

2. Time-Aware Processing:

.. code-block:: python

    # Process with variable time steps
    time_delta = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(batch_size, seq_len)
    )
    outputs = model(x, time_delta=time_delta)

3. State Management:

.. code-block:: python

    # Initialize states
    batch_size = 32
    initial_state = mx.zeros((batch_size, model.cell.units))
    
    # Process with explicit state
    outputs, final_state = model(x, initial_state=initial_state)

User's Guide
--------------

.. toctree::
    :maxdepth: 2

    quickstart
    mlx_guide
    apple_silicon_optimization
    performance_optimization
    examples/index
    api/index

Performance Guides
-----------------

.. toctree::
    :maxdepth: 1

    performance_guide
    advanced_profiling
    advanced_visualization
    testing_visualizations
