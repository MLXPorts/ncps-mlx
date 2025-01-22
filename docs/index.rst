:hide-toc:

===================================================
Welcome to Neural Circuit Policies's documentation!
===================================================

`Neural Circuit Policies (NCPs) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ are machine learning models inspired by the nervous system of the nematode *C. elegans*.
This package provides easy-to-use implementations of NCPs for PyTorch and Tensorflow.

.. code-block:: bash

    pip3 install -U ncps

Example Pytorch example:

.. code-block:: python

    from ncps.torch import CfC

    # a fully connected CfC network
    rnn = CfC(input_size=20, units=50)
    x = torch.randn(2, 3, 20) # (batch, time, features)
    h0 = torch.zeros(2,50) # (batch, units)
    output, hn = rnn(x,h0)

A Tensorflow example

.. code-block:: python

    # Tensorflow example
    from ncps.tf import LTC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4) # 28 neurons, 4 outputs
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            # LTC model with NCP sparse wiring
            LTC(wiring, return_sequences=True),
        ]
    )

Installation Guide
-------------------

To install the `ncps` package, follow these steps:

1. Ensure you have Python 3.6 or higher installed on your system.
2. Install the `ncps` package using pip:

.. code-block:: bash

    pip install ncps

3. Install the required dependencies:

.. code-block:: bash

    pip install torch tensorflow

4. Optionally, you can install additional dependencies for documentation and examples:

.. code-block:: bash

    pip install sphinx numpy sphinx-gallery numpydoc pandas loky tqdm distributed sphinx-copybutton furo sphinx_design

Examples and Use Cases
----------------------

Here are some examples and use cases for the different functionalities of the `ncps` package:

1. Creating a fully connected CfC network in PyTorch:

.. code-block:: python

    from ncps.torch import CfC

    rnn = CfC(input_size=20, units=50)
    x = torch.randn(2, 3, 20)
    h0 = torch.zeros(2, 50)
    output, hn = rnn(x, h0)

2. Creating an LTC model with NCP sparse wiring in TensorFlow:

.. code-block:: python

    from ncps.tf import LTC
    from ncps.wirings import AutoNCP

    wiring = AutoNCP(28, 4)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            LTC(wiring, return_sequences=True),
        ]
    )

3. Using the `ncps` package for time-series prediction:

.. code-block:: python

    import numpy as np
    from ncps.torch import CfC

    # Generate synthetic time-series data
    N = 100
    data_x = np.sin(np.linspace(0, 3 * np.pi, N)).reshape([1, N, 1])
    data_y = np.cos(np.linspace(0, 3 * np.pi, N)).reshape([1, N, 1])

    rnn = CfC(input_size=1, units=10, return_sequences=True)
    output = rnn(torch.tensor(data_x, dtype=torch.float32))

4. Combining NCPs with other layers in TensorFlow:

.. code-block:: python

    from ncps.tf import LTC
    from ncps.wirings import AutoNCP
    import tensorflow as tf

    wiring = AutoNCP(28, 4)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(None, 2)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (5, 5), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (5, 5), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu")),
            LTC(wiring, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax")),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='sparse_categorical_crossentropy')

User’s Guide
--------------

.. toctree::
    :maxdepth: 2

    quickstart
    examples/index
    api/index
