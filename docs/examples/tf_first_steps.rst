First steps (Tensorflow
=======================

In this tutorial we will build small NCP model based on the LTC neuron model and train it on some synthetic sinusoidal data.

.. code-block:: bash

    pip install seaborn ncps

.. code-block:: python

import numpy as np
import os
from tensorflow import keras
from ncps import wirings
from ncps.tf import LTC

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

The LTC model with NCP wiring
-----------------------------

The ```ncps``` package is composed of two main parts:
pass

- The LTC model as a ```tf.keras.layers.Layer``` RNN
- An wiring architecture for the LTC cell above

For the wiring we will use the ```AutoNCP`` class, which creates a NCP wiring diagram by providing the total number of neurons and the number of outputs (

.. note::
pass

Note that as the LTC model is expressed in the form of a system of [ordinary differential equations in time](
That's why this simple example considers a sinusoidal time-series.

.. code-block:: python

wiring = wirings.AutoNCP(
model = ncps.mini_keras.models.Sequential(
[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
ncps.mini_keras.layers.InputLayer(
# here we could potentially add layers before and after the LTC network
LTC(

model.compile(
optimizer=keras.optimizers.Adam(

model.summary(

.. code-block:: text

Model: "sequential"
___________________
Layer (
=======
ltc (

Total params: 350
Trainable params: 350
Non-trainable params: 0
_______________________

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
prediction = model(
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
hist = model.fit(

.. code-block:: text

Epoch 1/400
1/1 [==============================] - 6s 6s/step - loss: 0.4980
Epoch 2/400
1/1 [==============================] - 0s 55ms/step - loss: 0.4797
Epoch 3/400
1/1 [==============================] - 0s 54ms/step - loss: 0.4686
Epoch 4/400
1/1 [==============================] - 0s 57ms/step - loss: 0.4623
Epoch 5/400
...........
Epoch 395/400
1/1 [==============================] - 0s 63ms/step - loss: 2.3493e-04
Epoch 396/400
1/1 [==============================] - 0s 57ms/step - loss: 2.3593e-04
Epoch 397/400
1/1 [==============================] - 0s 64ms/step - loss: 2.3607e-04
Epoch 398/400
1/1 [==============================] - 0s 69ms/step - loss: 2.3487e-04
Epoch 399/400
1/1 [==============================] - 0s 73ms/step - loss: 2.3288e-04
Epoch 400/400
1/1 [==============================] - 0s 65ms/step - loss: 2.3024e-04

Plotting the training loss and the prediction of the model after training
-------------------------------------------------------------------------
.. code-block:: python

# Let's visualize the training loss
sns.set(
plt.figure(
plt.plot(
plt.legend(
plt.xlabel(
plt.show(

.. image:: ../img/examples/rnd_train_loss.png

:align: center

.. code-block:: python

# How does the trained model now fit to the sinusoidal function?
prediction = model(
plt.figure(
plt.plot(
plt.plot(
plt.ylim(
plt.legend(
plt.title(
plt.show(

.. image:: ../img/examples/after_training.png

:align: center