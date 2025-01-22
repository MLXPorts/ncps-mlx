:hide-toc:

===================================================
Quickstart
===================================================

Neural Circuit Policies are recurrent neural network models inspired by the nervous system of the nematode C. elegans.
Compared to standard ML models, NCPs have

#. Neurons that are modeled by an ordinary differential equation
#. A sparse structured wiring

Neuron Models
=============================
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
=============================
We can use both models described above with a fully-connected wiring diagram by simply passing the number of neurons, i.e., as it is done in standard ML models such as LSTMs, GRU, MLPs, and Transformers.

.. code-block:: python

    from ncps.torch import CfC

    # a fully connected CfC network
    rnn = CfC(input_size=20, units=50)

We can also specify sparse structured wirings in the form of a ``ncps.wirings.Wiring`` object.
The `Neural Circuit Policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ is the most interesting wiring paradigm provided in this package and comprises of a 4-layer recurrent connection principle of sensory, inter, command, and motor neurons.

.. image:: ./img/wirings.png
   :align: center

The easiest way to create a NCP wiring is via the ``AutoNCP`` class, which requires defining the total number of neurons and the number of motor neurons (= output size).

.. code-block:: python

    from ncps.torch import CfC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
    input_size = 20
    rnn = CfC(input_size, wiring)

Diagram
=============================

.. image:: ./img/things.png
   :align: center

Detailed Explanations and Examples
=============================

The `ncps` package provides various modules and functions to work with Neural Circuit Policies. Here are some detailed explanations and examples for each module and function:

1. `ncps.torch.CfC`: This module provides the implementation of the Closed-form Continuous-time (CfC) model for PyTorch. The CfC model approximates the closed-form solution of the differential equation, making it faster for training and inference compared to the LTC model.

.. code-block:: python

    from ncps.torch import CfC

    # Create a fully connected CfC network
    rnn = CfC(input_size=20, units=50)
    x = torch.randn(2, 3, 20)  # (batch, time, features)
    h0 = torch.zeros(2, 50)  # (batch, units)
    output, hn = rnn(x, h0)

2. `ncps.tf.LTC`: This module provides the implementation of the Liquid Time-Constant (LTC) model for TensorFlow. The LTC model is based on neurons in the form of differential equations interconnected via sigmoidal synapses.

.. code-block:: python

    from ncps.tf import LTC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            LTC(wiring, return_sequences=True),
        ]
    )

3. `ncps.wirings.AutoNCP`: This module provides an easy way to create a Neural Circuit Policy (NCP) wiring by specifying the total number of neurons and the number of motor neurons (output size).

.. code-block:: python

    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs

4. `ncps.datasets`: This module provides various datasets for training and evaluating Neural Circuit Policies. For example, the `AtariCloningDataset` class can be used to load Atari game data for training.

.. code-block:: python

    from ncps.datasets import AtariCloningDataset

    dataset = AtariCloningDataset(env_name="Pong", split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

5. `ncps.mini_keras`: This module provides a lightweight implementation of Keras-like layers and models for working with Neural Circuit Policies. It includes various layers, activations, and utilities for building and training models.

.. code-block:: python

    from ncps.mini_keras import layers, models

    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(None, 20)),
            layers.Dense(50, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

Troubleshooting
=============================

Here are some common issues and errors that users may encounter when using the `ncps` package, along with their solutions:

1. ImportError: No module named 'mlx'

   Solution: Ensure that the `mlx` package is installed. You can install it using pip:

.. code-block:: bash

    pip install mlx

2. AttributeError: 'LTCCell' object has no attribute 'call'

   Solution: Ensure that you are using the correct version of the `ncps` package. Update to the latest version if necessary:

.. code-block:: bash

    pip install --upgrade ncps

3. NameError: name 'backend' is not defined

   Solution: Ensure that the `backend` module is imported in your code. Add the following import statement at the beginning of your script:

.. code-block:: python

    import backend

4. ValueError: If sparsity of a CfC cell is set, then no backbone is allowed

   Solution: Ensure that the `backbone_units` parameter is set to 0 when using sparsity in a CfC cell. For example:

.. code-block:: python

    rnn = CfCCell(input_size=20, units=50, sparsity_mask=sparsity_mask, backbone_units=0)

5. RuntimeError: Running a CfC with mixed_memory=True requires a tuple (h0, c0) to be passed as state

   Solution: Ensure that you are passing a tuple (h0, c0) as the initial state when using mixed memory in a CfC model. For example:

.. code-block:: python

    h0 = torch.zeros(2, 50)
    c0 = torch.zeros(2, 50)
    output, hn = rnn(x, (h0, c0))
