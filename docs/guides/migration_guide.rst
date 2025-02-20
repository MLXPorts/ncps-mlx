Migration Guide
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
===============
==============

This guide helps you migrate your Neural Circuit Policies code to the MLX implementation, which is optimized for Apple Silicon.

Framework Differences
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
------------------

MLX vs PyTorch
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~

1. **Model Creation**

   PyTorch:
   
   .. code-block:: python

       from ncps.torch import CfC
       
       model = CfC(
           input_size=10,
           hidden_size=32,
           num_layers=2,
           bidirectional=True
       )

   MLX:
   
   .. code-block:: python

       from ncps.mlx import CfC, CfCCell
       from ncps.wirings import AutoNCP
       
       # Create wiring
       wiring = AutoNCP(units=32, output_size=4)
       
       # Create model
       model = CfC(
           cell=CfCCell(
               wiring=wiring,
               activation="tanh",
               backbone_units=[64, 64],
               backbone_layers=2
           ),
           bidirectional=True,
           merge_mode="concat"
       )

2. **Data Types**

   PyTorch:
   
   .. code-block:: python

       import torch
       
       x = torch.randn(32, 10, 8)
       x = x.to(torch.float32)

   MLX:
   
   .. code-block:: python

       import mlx.core as mx
       
       x = mx.random.normal((32, 10, 8))
       # MLX uses float32 by default

3. **Training Loop**

   PyTorch:
   
   .. code-block:: python

       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       
       for epoch in range(epochs):
           optimizer.zero_grad()
           output = model(x)
           loss = criterion(output, y)
           loss.backward()
           optimizer.step()

   MLX:
   
   .. code-block:: python

       optimizer = mx.optimizers.Adam(learning_rate=0.001)
       
       def loss_fn(model, x, y):
           pred = model(x)
           return mx.mean((pred - y) ** 2)
       
       loss_and_grad_fn = mx.value_and_grad(model, loss_fn)
       
       for epoch in range(epochs):
           loss, grads = loss_and_grad_fn(model, x, y)
           optimizer.update(model, grads)

MLX vs TensorFlow
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~

1. **Model Creation**

   TensorFlow:
   
   .. code-block:: python

       from ncps.tf import LTC
       
       model = tf.keras.Sequential([
           tf.keras.layers.InputLayer(input_shape=(None, 10)),
           LTC(units=32, return_sequences=True)
       ])

   MLX:
   
   .. code-block:: python

       from ncps.mlx import LTC, LTCCell
       from ncps.wirings import AutoNCP
       
       wiring = AutoNCP(units=32, output_size=4)
       
       model = LTC(
           cell=LTCCell(
               wiring=wiring,
               activation="tanh",
               backbone_units=[64]
           ),
           return_sequences=True
       )

2. **Data Processing**

   TensorFlow:
   
   .. code-block:: python

       import tensorflow as tf
       
       dataset = tf.data.Dataset.from_tensor_slices((x, y))
       dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

   MLX:
   
   .. code-block:: python

       import mlx.core as mx
       
       def create_batches(x, y, batch_size=32):
           indices = mx.random.permutation(len(x))
           for i in range(0, len(x), batch_size):
               batch_idx = indices[i:i + batch_size]
               yield x[batch_idx], y[batch_idx]

3. **Training**

   TensorFlow:
   
   .. code-block:: python

       model.compile(optimizer='adam', loss='mse')
       model.fit(dataset, epochs=10)

   MLX:
   
   .. code-block:: python

       optimizer = mx.optimizers.Adam(learning_rate=0.001)
       
       @mx.compile(static_argnums=(1,))
       def train_step(model, training=True):
           def loss_fn(model, x, y):
               pred = model(x, training=training)
               return mx.mean((pred - y) ** 2)
           return mx.value_and_grad(model, loss_fn)

Common Migration Tasks
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
-------------------

1. **State Management**

   .. code-block:: python

       # MLX state handling
       class StatefulModel(nn.Module):
           def __init__(self):
               super().__init__()
               self.cfc = CfC(
                   cell=CfCCell(wiring),
                   return_state=True
               )
               
           def __call__(self, x, state=None):
               return self.cfc(x, initial_state=state)

2. **Custom Layers**

   .. code-block:: python

       # MLX custom layer
       class CustomLayer(nn.Module):
           def __init__(self):
               super().__init__()
               self.linear = nn.Linear(10, 10)
               
           def __call__(self, x):
               return self.linear(x)

3. **Data Loading**

   .. code-block:: python

       # MLX data loading
       class DataLoader:
           def __init__(self, x, y, batch_size=32):
               self.x = mx.array(x)
               self.y = mx.array(y)
               self.batch_size = batch_size
               
           def __iter__(self):
               indices = mx.random.permutation(len(self.x))
               for i in range(0, len(self.x), self.batch_size):
                   batch_idx = indices[i:i + self.batch_size]
                   yield self.x[batch_idx], self.y[batch_idx]

Best Practices
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
------------

1. **Hardware Optimization**

   - Use MLX's lazy evaluation
   - Enable operator fusion
   - Optimize batch sizes
   - Monitor memory usage

2. **Code Structure**

   - Separate model definition
   - Use functional components
   - Implement proper state management
   - Handle device placement

3. **Performance**

   - Profile code sections
   - Use MLX compilation
   - Optimize memory usage
   - Monitor training metrics

Common Issues
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-----------

1. **Memory Management**

   - Clear unused variables
   - Use appropriate batch sizes
   - Monitor memory usage
   - Implement checkpointing

2. **Performance**

   - Enable MLX optimizations
   - Profile bottlenecks
   - Use efficient architectures
   - Monitor hardware utilization

3. **Training**

   - Implement proper logging
   - Monitor gradients
   - Track metrics
   - Validate results

Getting Help
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
----------

For migration assistance:

1. Check example notebooks
2. Review documentation
3. File GitHub issues
4. Join discussions

References
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
---------

- `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
- `PyTorch to MLX Guide <https://ml-explore.github.io/mlx/build/html/notebooks/pytorch_to_mlx.html>`_
- `TensorFlow to MLX Guide <https://ml-explore.github.io/mlx/build/html/notebooks/tensorflow_to_mlx.html>`_
- `Apple Silicon Optimization Guide <https://developer.apple.com/documentation/accelerate>`_
