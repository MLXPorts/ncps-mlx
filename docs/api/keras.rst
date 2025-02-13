.. keras-api-label:

Keras Layers
----------------------------

.. toctree::
    :maxdepth: 2

.. automodule:: ncps.keras

Sequence models
============================================

.. autoclass:: CfC
   :members:

.. autoclass:: LTC
   :members:

Single time-step models (RNN **cells**)
============================================

.. autoclass:: CfCCell
   :members:

.. autoclass:: LTCCell
   :members:

.. note::
   These layers are implemented using Keras 3.x and are backend-agnostic. They can work with TensorFlow, JAX, or PyTorch backends depending on your Keras configuration.