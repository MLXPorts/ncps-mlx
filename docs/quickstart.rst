:hide-toc:

Quickstart
==========

Neural Circuit Policies are recurrent neural network models inspired by the nervous system of the nematode C. elegans.
Compared to standard ML models, NCPs have

#. Neurons that are modeled by an ordinary differential equation
#. A sparse structured wiring

Neuron Models
=============
The package currently provides two neuron models: LTC and CfC:
The `liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936/16743>`_ model is based on neurons in the form of differential equations interconnected via sigmoidal synapses.
The term liquid time-constant comes from the property of LTCs that their timing behavior is adaptive to the input (how fast or slow they respond to some stimulus can depend on the specific input).
Because LTCs are ordinary differential equations, their behavior can only be described over time.
LTCs are universal approximators and implement causal dynamical models.
However, the LTC model has one major disadvantage: to compute their output, we need a numerical differential equation-solver which seriously slows down their training and inference time.
Closed-form continuous-time (CfC) models resolve this bottleneck by approximating the closed-form solution of the differential equation.

.. note::

    Both the LTC and the CfC models are **recurrent neural networks** and possess a temporal state. Therefore, these models are applicable only to sequential or time-series data.

Wirings
=======
Neural Circuit Policies use wiring patterns to define the connectivity between neurons. The package provides several wiring patterns:

.. code-block:: python

from ncps.wirings import AutoNCP, NCP, FullyConnected

# Automatic NCP wiring
wiring = AutoNCP(
units=32,          # Total number of neurons
output_size=4,     # Number of output neurons
sparsity_level=0.5 # Connection sparsity

# Manual NCP wiring
wiring = NCP(
inter_neurons=16,      # Number of interneurons
command_neurons=8,     # Number of command neurons
motor_neurons=4,       # Number of motor neurons
sensory_fanout=4,      # Sensory neuron connections
inter_fanout=4,        # Interneuron connections
recurrent_command_synapses=3,  # Recurrent connections
motor_fanin=4          # Motor neuron inputs

# Fully connected wiring
wiring = FullyConnected(
    units=32,
output_size=4

.. image:: ./img/wirings.png

:align: center

Creating Models
===============
Models can be created using the wiring patterns above. Here's how to create models with different frameworks:
pass

MLX Implementation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

import mlx.core as mx
from ncps.mlx import CfC, CfCCell, LTC, LTCCell
from ncps.wirings import AutoNCP

# Create wiring
wiring = AutoNCP(

# Create CfC model
cfc_model = CfC(
cell=CfCCell(
    wiring=wiring,
        activation="tanh",
        backbone_units=[64, 64],
    backbone_layers=2
        ),
            return_sequences=True,
                bidirectional=True,
            merge_mode="concat"

            # Create LTC model
            ltc_model = LTC(
            cell=LTCCell(
                wiring=wiring,
                    activation="tanh",
                    backbone_units=[64],
                backbone_layers=1
                    ),
                return_sequences=False

                # Process sequence
                x = mx.random.normal(
                time_delta = mx.ones(
                outputs, states = cfc_model(

            PyTorch Implementation
            ~~~~~~~~~~~~~~~~~~~~~~

            .. code-block:: python

            from ncps.torch import CfC
            from ncps.wirings import AutoNCP

            wiring = AutoNCP(
            model = CfC(

        Keras Implementation
        ~~~~~~~~~~~~~~~~~~~~

        .. code-block:: python

        from ncps.keras import LTC
        from ncps.wirings import AutoNCP
        import keras

        wiring = AutoNCP(
        model = keras.Sequential(
        keras.layers.Input(
        LTC(

    Troubleshooting
    ===============

    Common Issues
    ~~~~~~~~~~~~~

    1. ImportError: No module named 'mlx'

    Solution: Install MLX package:
    pass

    .. code-block:: bash

    pip install mlx

    2. ValueError: Expected input dimension mismatch

    Solution: Ensure input shapes match:

    .. code-block:: python

    # Correct shapes
    x = mx.random.normal(
    time_delta = mx.ones(

3. ValueError: Wiring must be built before use

Solution: Build wiring with input dimension:

.. code-block:: python

wiring = AutoNCP(
wiring.build(

For basic examples, see:

- examples/notebooks/mlx_cfc_example.ipynb
- examples/notebooks/mlx_ltc_rnn_example.ipynb

For advanced features and usage, see:

- docs/deepdive/advanced_features.rst

