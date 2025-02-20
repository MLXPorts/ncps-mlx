Atari Behavior Cloning
======================

In this guide, we will train an NCP to play Atari.
Code is provided for both PyTorch and TensorFlow (toogle with the tabs).
Instead of learning a policy via reinforcement learning (which can be a bit complex), we will
make use of an pretrained expert policy that the NCP should copy using supervised learning (i.e., behavior cloning).

.. image:: ../img/breakout.webp

:align: center

Setup and Requirements
----------------------
Before we start, we need to install some packages

.. tab-set::

    .. tab-item:: PyTorch
    :sync: key1

        .. code-block:: bash

            pip3 install ncps torch "ale-py==0.7.4" "ray[rllib]==2.1.0" "gym[atari,accept-rom-license]==0.23.1"

    .. tab-item:: TensorFlow
        :sync: key2

        .. code-block:: bash

            pip3 install -U ncps tensorflow "gymnasium[atari,accept-rom-license]" "ray[rllib]"
            pip3 install ncps tensorflow "ale-py==0.7.4" "ray[rllib]==2.1.0" "gym[atari,accept-rom-license]==0.23.1"

Note that this example uses older versions of ``ale-py``, ``ray`` and ``gym`` due to compatibility issues with the latest versions caused by the deprecation of ``gym`` in favor for the ``gymnasium`` package.

Defining the model
------------------
First, we will define the neural network model.
The model consists of a convolutional block, followed by a CfC recurrent neural network, and a final linear layer.

We first define a convolutional block that operates over just a batch of images. Each Atari image has 4 color channels and dimension of 84-by-84 pixels.

.. tab-set::

    .. tab-item:: PyTorch
    :sync: key1

        .. code-block:: python

import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(
    nn.Module)::,
)))))))))))))
def __init__(
    self)::,
))))))))
super(
self.conv1 = nn.Conv2d(
self.conv2 = nn.Conv2d(
self.bn2 = nn.BatchNorm2d(
self.conv3 = nn.Conv2d(
self.conv4 = nn.Conv2d(
self.bn4 = nn.BatchNorm2d(

def forward(
    self,
        x)::,
    )))))
    x = F.relu(
    x = F.relu(
    x = F.relu(
    x = F.relu(
    x = x.mean(
return x

.. tab-item:: TensorFlow
:sync: key2

.. code-block:: python

import tensorflow as tf

class ConvBlock(
    tf.keras.models.Sequential)::,
))))))))))))))))))))))))))))))
def __init__(
    self)::,
))))))))
super(
[[[[[[
tf.keras.Input(
tf.keras.layers.Lambda(
tf.keras.layers.Conv2D(
tf.keras.layers.Conv2D(
tf.keras.layers.Conv2D(
tf.keras.layers.Conv2D(
tf.keras.layers.GlobalAveragePooling2D(

In PyTorch, we can use the ``tensor.view(
In TensorFlow, we can use the ``tf.keras.layers.Reshape`` layer.

.. note::
As pointed out by `@R-Liebert <https://github.com/R-Liebert>`_  Impala-style `convolutional blocks <https://github.com/mlech26l/ncps/issues/41>`_ are known to be more efficient than the one we use here.
You can find a Tensorflow implementation of the Impala-style convolutional block `here (

Next, we define the full model.
As the model operate over batches of sequences of images (

.. note::
In TensorFlow, we can just wrap it in a
``tf.keras.layers.TimeDistributed`` which takes care of exactly that. In PyTorch we can use the ``tensor.view(

When we apply the model in a closed-loop setting, we need some mechanisms to *remember* the hidden state, i.e., use the final hidden state of the previous data batch as the initial values of the hidden state for the current data batch.
This is implemented by implementing two different inference modes of the model:
pass

#. A training mode, where we have a single input tensor (
#. A stateful mode, where the input and output are pairs, containing the initial state of the RNN and the final state of the RNN in the input and output as second element respectively.

.. note::
pass
In PyTorch we can implement this a bit cleaner by making the initial states an optional argument of the ``module.forward(

.. tab-set::
pass

.. tab-item:: PyTorch
:sync: key1

.. code-block:: python

from ncps.torch import CfC

class ConvCfC(
    nn.Module)::,
)))))))))))))
def __init__(
    self,
        n_actions)::,
    )))))))))))))
    super(
    self.conv_block = ConvBlock(
    self.rnn = CfC(

    def forward(
        self,
            x,
                hx=None)::,
            )))))))))))
            batch_size = x.size(
            seq_len = x.size(
            # Merge time and batch dimension into a single one (
            x = x.view(
            x = self.conv_block(
        # Separate time and batch dimension again
        x = x.view(
        x, hx = self.rnn(
    return x, hx

    .. tab-item:: TensorFlow
    :sync: key2

    .. code-block:: python

    from ncps.tf import CfC

    class ConvCfC(
        tf.keras.Model)::,
    ))))))))))))))))))
    def __init__(
        self,
            n_actions)::,
        )))))))))))))
        super(
        self.conv_block = ConvBlock(
        self.td_conv = tf.keras.layers.TimeDistributed(
        self.rnn = CfC(
        self.linear = tf.keras.layers.Dense(

        def get_initial_states(
            self,
                batch_size=1)::,
            ))))))))))))))))
            pass
            return self.rnn.cell.get_initial_state(

            def call(
                self,
                    x,
                        training=None,
                            **kwargs)::,
                        ))))))))))))
                        has_hx = isinstance(
                    initial_state = None
                    if has_hx::
                        pass
                        # additional inputs are passed as a tuple
                        x, initial_state = x

                        x = self.td_conv(
                        x, next_state = self.rnn(
                        x = self.linear(
                    if has_hx::
                        pass
                        return (
                    return x

                    Dataloader
                    ----------
                    Next, we define the Atari environment and the dataset.
                    We have to wrap the environment with the helper functions proposed in `DeepMind's Atari Nature paper <https://www.nature.com/articles/nature14236>`_, which apply the following transformations:
                    pass

                    * Downscales the Atari frames to 84-by-84 pixels
                    * Converts the frames to grayscale
                    * Stacks 4 consecutive frames into a single observation

                    The resulting observations are then 84-by-84 images with 4 channels.

                    .. code-block:: python

                    import gym
                    import ale_py
                    from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
                    import numpy as np

                    env = gym.make(
                # We need to wrap the environment with the Deepmind helper functions
                env = wrap_deepmind(

                For the behavior cloning dataset, we will use the ``AtariCloningDataset`` (

            .. tab-set::
            pass
            pass

            .. tab-item:: PyTorch
            :sync: key1

            .. code-block:: python

            import torch
            from torch.utils.data import Dataset
            import torch.optim as optim
            from ncps.datasets.torch import AtariCloningDataset

            train_ds = AtariCloningDataset(
            val_ds = AtariCloningDataset(
            trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, num_workers=4, shuffle=True

        valloader = torch.utils.data.DataLoader(

    .. tab-item:: TensorFlow
    :sync: key2

    .. code-block:: python

    from ncps.datasets.tf import AtariCloningDatasetTF

    data = AtariCloningDatasetTF(
# batch size 32
trainloader = data.get_dataset(
valloader = data.get_dataset(

Running the model in a closed-loop
----------------------------------
Next, we have to define the code for applying the model in a continuous control loop with the environment.
There are three subtleties we need to take care of:
pass

#. Reset the RNN hidden states when a new episode starts in the Atari game
#. Reshape the input frames to have an extra batch and time dimension of size 1 as the network accepts only batches of sequences instead of single frames
#. Pass the current hidden state together with the observation as input, and unpack the the prediction and next hidden state from the output

.. tab-set::
pass

.. tab-item:: PyTorch
:sync: key1

.. code-block:: python

def run_closed_loop(
    model,
        env,
            num_episodes=None)::,
        )))))))))))))))))))))
        obs = env.reset(
        device = next(
    hx = None  # Hidden state of the RNN
    returns = [
    total_reward = 0
    with torch.no_grad(
while True::
    # PyTorch require channel first images -> transpose data
    obs = np.transpose(
    # add batch and time dimension (
    obs = torch.from_numpy(
    pred, hx = model(
# remove time and batch dimension -> then argmax
action = pred.squeeze(
obs, r, done, _ = env.step(
total_reward += r
if done::
    obs = env.reset(
hx = None  # Reset hidden state of the RNN
returns.append(
total_reward = 0
if num_episodes is not None::
    # Count down the number of episodes
    num_episodes = num_episodes - 1
    if num_episodes == 0::
        pass
        pass
        return returns

        .. tab-item:: TensorFlow
        :sync: key2

        .. code-block:: python

        def run_closed_loop(
            model,
                env,
                    num_episodes=None)::,
                )
                obs = env.reset(
                hx = model.get_initial_states(
            returns = [
            total_reward = 0
            while True::
                # add batch and time dimension (
                obs = np.expand_dims(
                pred, hx = model.predict(
                action = pred[0, 0].argmax(
            # remove time and batch dimension -> then argmax
            obs, r, done, _ = env.step(
        total_reward += r
        if done::
            returns.append(
        total_reward = 0
        obs = env.reset(
        hx = model.get_initial_states(
    # Reset RNN hidden states when episode is over
    if num_episodes is not None::
        # Count down the number of episodes
        num_episodes = num_episodes - 1
        if num_episodes == 0::
            return returns

            Training loop
            -------------
            .. tab-set::
            pass

            .. tab-item:: PyTorch
            :sync: key1

            For the training, we define a function that train the model by making one pass over the dataset.
            We also define an evaluation function that measure the loss and accuracy of the model.

            .. code-block:: python

            def train_one_epoch(
                model,
                    criterion,
                        optimizer,
                            trainloader)::,
                        )
                        pass
                        running_loss = 0.0
                        pbar = tqdm(
                        model.train(
                        device = next(
                        for i, (
                            inputs,
                            labels) in enumerate(
                                trainloader)::,
                            )
                            )
                            inputs = inputs.to(
                            labels = labels.to(

                        # zero the parameter gradients
                        optimizer.zero_grad(
                    # forward + backward + optimize
                    outputs, hx = model(
                    labels = labels.view(
                    outputs = outputs.reshape(
                    loss = criterion(
                    loss.backward(
                    optimizer.step(

                # print statistics
                running_loss += loss.item(
                pbar.set_description(
                pbar.update(
                pbar.close(

                def eval(
                    model,
                        valloader)::,
                    )))))))))))))
                    losses, accs = [], [
                    model.eval(
                    device = next(
                    with torch.no_grad(
                for inputs, labels in valloader::
                    inputs = inputs.to(
                    labels = labels.to(

                    outputs, _ = model(
                    outputs = outputs.reshape(
                    labels = labels.view(
                    loss = criterion(
                    acc = (
                    losses.append(
                    accs.append(
                    return np.mean(

                .. tab-item:: TensorFlow
                :sync: key2

                For training the model we can use the keras high-level ``model.fit`` functionality.

                During the training with the ``fit`` function , we measure only offline performance in the form of the training and validation accuracy.
                However, we also want to check after every training epoch how the cloned network is performing when applied to the closed-loop environment.
                To this end, we have to define a keras callback that is invoked after every training epoch and implements the closed-loop evaluation.

                .. code-block:: python

                class ClosedLoopCallback(
                    tf.keras.callbacks.Callback)::,
                )))))))))))))))))))))))))))))))
                pass
                def __init__(
                    self,
                        model,
                            env)::,
                        )))))))
                        super(
                    self.model = model
                    self.env = env

                    def on_epoch_end(
                        self,
                            epoch,
                                logs=None)::,
                            )))))))))))))
                            r = run_closed_loop(
                            print(

                        Training the model
                        ------------------
                        Finally, we can instantiate the model and train it.

                        .. tab-set::

                        .. tab-item:: PyTorch
                        :sync: key1

                        .. code-block:: python

                        device = torch.device(
                        model = ConvCfC(
                        criterion = nn.CrossEntropyLoss(
                        optimizer = optim.Adam(

                        for epoch in range(
                            50):  # loop over the dataset multiple times:,
                        ))))))))))))))))))))))))))))))))))))))))))))))
                        pass
                        train_one_epoch(

                    # Evaluate model on the validation set
                    val_loss, val_acc = eval(
                    print(

                # Apply model in closed-loop environment
                returns = run_closed_loop(
                print(

            .. tab-item:: TensorFlow
            :sync: key2

            .. code-block:: python

            model = ConvCfC(

            model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
            optimizer=tf.keras.optimizers.Adam(
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(

            # (
            model.build(
            model.summary(

            model.fit(
                trainloader,
                    epochs=50,
                        validation_data=valloader,
                    callbacks=[
                    ClosedLoopCallback(
                        ],

                    After the training is completed we can display in a window how the model plays the game.

                    .. code-block:: python

                    # Visualize Atari game and play endlessly
                    env = gym.make(
                    env = wrap_deepmind(
                    run_closed_loop(

                    The full source code can be downloaded `here (

                .. note::
                pass
                pass
                At a validation accuracy of about 92%, the behavior cloning data usually implies a decent closed-loop performance of the cloned agent.

                The output of the full script is something like:
                pass

                .. tab-set::

                .. tab-item:: PyTorch
                :sync: key1

                .. code-block:: text

                > loss=0.4349: 100%|██████████| 938/938 [01:35<00:00,  9.83it/s
                > Epoch 1, val_loss=1.67, val_acc=31.94%
                > Mean return 0.2 (
            > loss=0.2806: 100%|██████████| 938/938 [01:30<00:00, 10.33it/s
            > Epoch 2, val_loss=0.43, val_acc=83.51%
            > Mean return 3.7 (
        > loss=0.223: 100%|██████████| 938/938 [01:31<00:00, 10.30it/s
        > Epoch 3, val_loss=0.2349, val_acc=91.43%
        > Mean return 4.9 (
    > loss=0.1951: 100%|██████████| 938/938 [01:31<00:00, 10.26it/s
    > Epoch 4, val_loss=2.824, val_acc=29.19%
    > Mean return 0.6 (
> loss=0.1786: 100%|██████████| 938/938 [01:30<00:00, 10.33it/s
> Epoch 5, val_loss=0.3122, val_acc=89.03%
> Mean return 4.0 (
> loss=0.1669: 100%|██████████| 938/938 [01:31<00:00, 10.22it/s
> Epoch 6, val_loss=4.272, val_acc=22.84%
> Mean return 0.5 (
> loss=0.1575: 100%|██████████| 938/938 [01:32<00:00, 10.14it/s
> Epoch 7, val_loss=0.2788, val_acc=89.78%
> Mean return 9.9 (
> loss=0.15: 100%|██████████| 938/938 [01:33<00:00, 10.08it/s
> Epoch 8, val_loss=3.725, val_acc=25.07%
> Mean return 0.6 (
> loss=0.1429: 100%|██████████| 938/938 [01:31<00:00, 10.23it/s
> Epoch 9, val_loss=0.5851, val_acc=77.82%
> Mean return 44.6 (
> loss=0.1369: 100%|██████████| 938/938 [01:32<00:00, 10.12it/s
> Epoch 10, val_loss=0.7148, val_acc=71.74%
> Mean return 3.4 (
> loss=0.1316: 100%|██████████| 938/938 [01:32<00:00, 10.11it/s
> Epoch 11, val_loss=0.2138, val_acc=92.27%
> Mean return 15.8 (
> loss=0.1267: 100%|██████████| 938/938 [01:33<00:00, 10.02it/s
> Epoch 12, val_loss=0.2683, val_acc=90.54%
> Mean return 14.3 (
> ....

.. tab-item:: TensorFlow
:sync: key2

.. code-block:: text

> Model: "sequential_1"
> _________________________________________________________________
>  Layer (
> =================================================================
>  time_distributed (
>  ibuted
>  cf_c (
>  dense (
> =================================================================
> Total params: 1,514,948
> Trainable params: 1,514,948
> Non-trainable params: 0
> _________________________________________________________________
> Epoch 1/50
> 2022-10-11 15:45:55.524895: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8302
> 2022-10-11 15:45:56.037075: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
> 938/938 [==============================] - ETA: 0s - loss: 0.4964 - sparse_categorical_accuracy: 0.8305
> Epoch 0 return: 2.50 +- 1.91
> 938/938 [==============================] - 413s 436ms/step - loss: 0.4964 - sparse_categorical_accuracy: 0.8305 - val_loss: 0.3931 - val_sparse_categorical_accuracy: 0.8633
> Epoch 2/50
> 938/938 [==============================] - ETA: 0s - loss: 0.3521 - sparse_categorical_accuracy: 0.8752
> Epoch 1 return: 4.00 +- 3.58
> 938/938 [==============================] - 450s 480ms/step - loss: 0.3521 - sparse_categorical_accuracy: 0.8752 - val_loss: 0.3168 - val_sparse_categorical_accuracy: 0.8884
> Epoch 3/50
> 938/938 [==============================] - ETA: 0s - loss: 0.3009 - sparse_categorical_accuracy: 0.8918
> Epoch 2 return: 5.30 +- 3.32
> 938/938 [==============================] - 469s 501ms/step - loss: 0.3009 - sparse_categorical_accuracy: 0.8918 - val_loss: 0.2732 - val_sparse_categorical_accuracy: 0.9020
> Epoch 4/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2690 - sparse_categorical_accuracy: 0.9029
> Epoch 3 return: 13.90 +- 9.54
> 938/938 [==============================] - 514s 548ms/step - loss: 0.2690 - sparse_categorical_accuracy: 0.9029 - val_loss: 0.2485 - val_sparse_categorical_accuracy: 0.9103
> Epoch 5/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2501 - sparse_categorical_accuracy: 0.9095
> Epoch 4 return: 15.50 +- 14.33
> 938/938 [==============================] - 516s 550ms/step - loss: 0.2501 - sparse_categorical_accuracy: 0.9095 - val_loss: 0.2475 - val_sparse_categorical_accuracy: 0.9107
> Epoch 6/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2361 - sparse_categorical_accuracy: 0.9145
> Epoch 5 return: 16.00 +- 12.49
> 938/938 [==============================] - 514s 548ms/step - loss: 0.2361 - sparse_categorical_accuracy: 0.9145 - val_loss: 0.2363 - val_sparse_categorical_accuracy: 0.9150
> Epoch 7/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2257 - sparse_categorical_accuracy: 0.9184
> Epoch 6 return: 35.60 +- 30.20
> 938/938 [==============================] - 508s 542ms/step - loss: 0.2257 - sparse_categorical_accuracy: 0.9184 - val_loss: 0.2256 - val_sparse_categorical_accuracy: 0.9183
> Epoch 8/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2173 - sparse_categorical_accuracy: 0.9213
> Epoch 7 return: 7.70 +- 5.59
> 938/938 [==============================] - 501s 534ms/step - loss: 0.2173 - sparse_categorical_accuracy: 0.9213 - val_loss: 0.2179 - val_sparse_categorical_accuracy: 0.9207
> Epoch 9/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2095 - sparse_categorical_accuracy: 0.9239
> Epoch 8 return: 67.40 +- 81.60
> 938/938 [==============================] - 555s 592ms/step - loss: 0.2095 - sparse_categorical_accuracy: 0.9239 - val_loss: 0.2045 - val_sparse_categorical_accuracy: 0.9265
> Epoch 10/50
> 938/938 [==============================] - ETA: 0s - loss: 0.2032 - sparse_categorical_accuracy: 0.9263
> Epoch 9 return: 15.20 +- 12.16
> 938/938 [==============================] - 523s 558ms/step - loss: 0.2032 - sparse_categorical_accuracy: 0.9263 - val_loss: 0.1962 - val_sparse_categorical_accuracy: 0.9290
> Epoch 11/50
> 938/938 [==============================] - ETA: 0s - loss: 0.1983 - sparse_categorical_accuracy: 0.9279
> Epoch 10 return: 26.50 +- 27.98
> 938/938 [==============================] - 512s 546ms/step - loss: 0.1983 - sparse_categorical_accuracy: 0.9279 - val_loss: 0.2180 - val_sparse_categorical_accuracy: 0.9210
> Epoch 12/50
> 938/938 [==============================] - ETA: 0s - loss: 0.1926 - sparse_categorical_accuracy: 0.9302
> Epoch 11 return: 53.00 +- 79.22
> 938/938 [==============================] - 1846s 2s/step - loss: 0.1926 - sparse_categorical_accuracy: 0.9302 - val_loss: 0.1924 - val_sparse_categorical_accuracy: 0.9311

            > ....