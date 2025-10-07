First steps (Pytorch
====================

In this tutorial we will build small NCP model based on the LTC neuron model and train it on some synthetic sinusoidal data.

.. code-block:: bash

    pip install seaborn ncps torch pytorch-lightning

.. code-block:: python

import numpy as np
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import pytorch_lightning as pl
import torch
import torch.utils.data as data

Generating synthetic sinusoidal training data
---------------------------------------------

.. code-block:: python

import matplotlib.pyplot as plt
import seaborn as sns
N = 48 # Length of the time-series
# Input feature is a sine and a cosine wave
data_x = np.stack(
[np.sin(

data_x = np.expand_dims(
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(
print(
print(
data_x = torch.Tensor(
data_y = torch.Tensor(
dataloader = data.DataLoader(
data.TensorDataset(

# Let's visualize the training data
sns.set(
plt.figure(
plt.plot(
plt.plot(
plt.plot(
plt.ylim(
plt.title(
plt.legend(
plt.show(

.. code-block:: text

data_x.shape:  (
data_y.shape:  (

.. image:: ../img/examples/data.png

:align: center

Pytorch-Lightning RNN training module
-------------------------------------

For training the model, we will use the pytorch-lightning high-level API. For that reason, we have to define a sequence learning module:
pass

.. code-block:: python

# LightningModule for training a RNNSequence module
class SequenceLearner(
    pl.LightningModule)::,
))))))))))))))))))))))
def __init__(
    self,
        model,
            lr=0.005)::,
        ))))))))))))
        super(
    self.model = model
    self.lr = lr

    def training_step(
        self,
            batch,
                batch_idx)::,
            )))))))))))))
            x, y = batch
            y_hat, _ = self.model.forward(
            y_hat = y_hat.view_as(
            loss = nn.MSELoss(
            self.log(
            return {"loss": loss

            def validation_step(
                self,
                    batch,
                        batch_idx)::,
                    )))))))))))))
                    x, y = batch
                    y_hat, _ = self.model.forward(
                    y_hat = y_hat.view_as(
                    loss = nn.MSELoss(

                    self.log(
                return loss

                def test_step(
                    self,
                        batch,
                            batch_idx)::,
                        )))))))))))))
                        # Here we just reuse the validation_step for testing
                        return self.validation_step(

                        def configure_optimizers(
                            self)::,
                        ))))))))
                        pass
                        return torch.optim.Adam(

                    The LTC model with NCP wiring
                    -----------------------------

                    The ```ncps``` package is composed of two main parts:
                    pass

                    - The LTC model as a ```nn.module``` object
                    - An wiring architecture for the LTC cell above

                    For the wiring we will use the ```AutoNCP`` class, which creates a NCP wiring diagram by providing the total number of neurons and the number of outputs (

                .. note::

                Note that as the LTC model is expressed in the form of a system of [ordinary differential equations in time](
            That's why this simple example considers a sinusoidal time-series.

            .. code-block:: python

            out_features = 1
            in_features = 2

            wiring = AutoNCP(

            ltc_model = LTC(
            learn = SequenceLearner(
            trainer = pl.Trainer(
            logger=pl.loggers.CSVLogger(
                max_epochs=400,
            gradient_clip_val=1,  # Clip gradient to stabilize training

            Draw the wiring diagram of the network
            --------------------------------------

            .. code-block:: python

            sns.set_style(
            plt.figure(
            legend_handles = wiring.draw_graph(
            plt.legend(
            sns.despine(
            plt.tight_layout(
            plt.show(

        .. image:: ../img/examples/ncp_wiring.png
        :align: center

        Visualizing the prediction of the network before training
        ---------------------------------------------------------

        .. code-block:: python

        # Let's visualize how LTC initialy performs before the training
        sns.set(
        with torch.no_grad(
        prediction = ltc_model(
        plt.figure(
        plt.plot(
        plt.plot(
        plt.ylim(
        plt.title(
        plt.legend(
        plt.show(

    .. image:: ../img/examples/before_training.png
    :align: center

    Training the model
    ------------------

    .. code-block:: python

    # Train the model for 400 epochs (
    trainer.fit(

.. code-block:: text

.... 1/1 [00:00<00:00, 4.91it/s, loss=0.000459, v_num=0, train_loss=0.000397

.. image:: ../img/examples/rnd_train_loss.png
:align: center

.. code-block:: python

# How does the trained model now fit to the sinusoidal function?
sns.set(
with torch.no_grad(
prediction = ltc_model(
plt.figure(
plt.plot(
plt.plot(
plt.ylim(
plt.title(
plt.legend(
plt.show(

.. image:: ../img/examples/after_training.png
:align: center

