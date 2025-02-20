Troubleshooting Guide
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
=====================
==================

This guide helps you diagnose and resolve common issues when working with Neural Circuit Policies in MLX.

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

Memory Issues
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~

1. **Out of Memory Errors**

   Symptom:
   
   .. code-block:: text
   
       RuntimeError: Out of memory

   Solutions:
   
   a. Reduce batch size:
   
      .. code-block:: python
      
          # Instead of
          model(large_batch)
          
          # Try
          batch_size = 32
          for i in range(0, len(data), batch_size):
              model(data[i:i+batch_size])

   b. Use gradient accumulation:
   
      .. code-block:: python
      
          accumulated_grads = None
          for micro_batch in data:
              loss, grads = loss_and_grad_fn(model, micro_batch)
              if accumulated_grads is None:
                  accumulated_grads = grads
              else:
                  for k in grads:
                      accumulated_grads[k] += grads[k]
          
          # Scale and apply
          for k in accumulated_grads:
              accumulated_grads[k] /= len(data)
          optimizer.update(model, accumulated_grads)

2. **Memory Leaks**

   Symptom:
   
   Gradually increasing memory usage over time.

   Solutions:
   
   a. Clear unused variables:
   
      .. code-block:: python
      
          import gc
          
          def train_epoch():
              for batch in data:
                  # Process batch
                  del intermediate_results  # Clear unused variables
                  gc.collect()  # Force garbage collection

   b. Use context managers for large operations:
   
      .. code-block:: python
      
          class MemoryContext:
              def __enter__(self):
                  return self
              
              def __exit__(self, *args):
                  gc.collect()
          
          with MemoryContext():
              large_operation()

Performance Issues
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~

1. **Slow Training**

   Symptom:
   
   Training is significantly slower than expected.

   Solutions:
   
   a. Use lazy evaluation effectively:
   
      .. code-block:: python
      
          # Bad: Eager evaluation
          for batch in data:
              loss = compute_loss(model, batch)
              mx.eval(loss)  # Unnecessary evaluation
          
          # Good: Lazy evaluation
          losses = []
          for batch in data:
              losses.append(compute_loss(model, batch))
          mx.eval(losses)  # Evaluate once at the end

   b. Optimize backbone configuration:
   
      .. code-block:: python
      
          # More efficient configuration
          model = CfC(
              input_size=10,
              hidden_size=32,
              backbone_units=64,  # Power of 2
              backbone_layers=2   # Balance depth vs speed
          )

2. **GPU Underutilization**

   Symptom:
   
   Low GPU utilization during training.

   Solutions:
   
   a. Increase batch size:
   
      .. code-block:: python
      
          # Find optimal batch size
          def find_optimal_batch_size(start_size=32):
              for size in [start_size * 2**i for i in range(5)]:
                  try:
                      train_batch(size)
                  except:
                      return size // 2
              return size

   b. Use compiled functions:
   
      .. code-block:: python
      
          @mx.compile
          def training_step(model, x, y):
              return model(x, y)

Numerical Issues
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~

1. **NaN Values**

   Symptom:
   
   Training loss becomes NaN.

   Solutions:
   
   a. Add gradient clipping:
   
      .. code-block:: python
      
          def clip_gradients(grads, max_norm=1.0):
              total_norm = mx.sqrt(sum(mx.sum(g**2) for g in grads.values()))
              clip_coef = max_norm / (total_norm + 1e-6)
              if clip_coef < 1:
                  for k in grads:
                      grads[k] *= clip_coef
              return grads
          
          # In training loop
          loss, grads = loss_and_grad_fn(model, batch)
          grads = clip_gradients(grads)
          optimizer.update(model, grads)

   b. Check for numerical stability:
   
      .. code-block:: python
      
          def stable_loss(pred, target):
              # Add epsilon for numerical stability
              return mx.mean((pred - target) ** 2 + 1e-6)

2. **Exploding Gradients**

   Symptom:
   
   Very large loss values or model weights.

   Solutions:
   
   a. Use gradient scaling:
   
      .. code-block:: python
      
          scale = 1.0 / batch_size
          grads = {k: g * scale for k, g in grads.items()}

   b. Initialize weights properly:
   
      .. code-block:: python
      
          model = CfC(
              input_size=10,
              hidden_size=32,
              initializer=nn.init.glorot_uniform
          )

Time-Aware Processing Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Incorrect Time Delta Shapes**

   Symptom:
   
   Shape mismatch errors with time_delta.

   Solutions:
   
   a. Check time delta shape:
   
      .. code-block:: python
      
          def check_time_delta(x, time_delta):
              if time_delta is not None:
                  expected_shape = (x.shape[0], x.shape[1], 1)
                  assert time_delta.shape == expected_shape, \
                      f"Expected shape {expected_shape}, got {time_delta.shape}"

   b. Reshape time delta properly:
   
      .. code-block:: python
      
          # Ensure correct shape
          if len(time_delta.shape) == 1:
              time_delta = time_delta.reshape(-1, 1, 1)

2. **Time Scale Issues**

   Symptom:
   
   Poor performance with variable time steps.

   Solutions:
   
   a. Normalize time deltas:
   
      .. code-block:: python
      
          def normalize_time(time_delta):
              return (time_delta - mx.mean(time_delta)) / (mx.std(time_delta) + 1e-6)

   b. Use log time scaling:
   
      .. code-block:: python
      
          def scale_time(time_delta):
              return mx.log1p(time_delta)  # log(1 + x) for numerical stability

Model-Specific Issues
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~

1. **CfC-Specific Issues**

   Symptom:
   
   Poor performance with CfC models.

   Solutions:
   
   a. Check mode configuration:
   
      .. code-block:: python
      
          model = CfC(
              input_size=10,
              hidden_size=32,
              mode="default",  # Try different modes
              activation="lecun_tanh"  # Use appropriate activation
          )

   b. Adjust backbone configuration:
   
      .. code-block:: python
      
          model = CfC(
              input_size=10,
              hidden_size=32,
              backbone_units=64,
              backbone_layers=2,
              backbone_dropout=0.1
          )

2. **LTC-Specific Issues**

   Symptom:
   
   Poor performance with LTC models.

   Solutions:
   
   a. Adjust time constant initialization:
   
      .. code-block:: python
      
          model = LTC(
              input_size=10,
              hidden_size=32,
              initializer=nn.init.uniform(-0.1, 0.1)
          )

   b. Use appropriate activation:
   
      .. code-block:: python
      
          model = LTC(
              input_size=10,
              hidden_size=32,
              activation="tanh"  # LTC works well with tanh
          )

Debugging Tips
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

1. **Gradient Checking**

   .. code-block:: python

    def check_gradients(model, x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        for k, g in grads.items():
            if mx.isnan(g).any():
                print(f"NaN gradient in {k}")
            if mx.isinf(g).any():
                print(f"Inf gradient in {k}")

2. **Model State Inspection**

   .. code-block:: python

    def inspect_model(model):
        state = model.state_dict()
        for k, v in state.items():
            if isinstance(v, mx.array):
                print(f"{k}: shape={v.shape}, mean={mx.mean(v)}, std={mx.std(v)}")

3. **Training Progress Monitoring**

   .. code-block:: python

    class ProgressMonitor:
        def __init__(self, window_size=100):
            self.losses = []
            self.window_size = window_size
            
        def update(self, loss):
            self.losses.append(float(loss))
            if len(self.losses) > self.window_size:
                self.losses.pop(0)
                
        def get_stats(self):
            return {
                'mean': np.mean(self.losses),
                'std': np.std(self.losses),
                'min': np.min(self.losses),
                'max': np.max(self.losses)
            }

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

If you encounter issues not covered in this guide:

1. Check the example notebooks
2. Review the API documentation
3. Run the test suite
4. File an issue on GitHub
5. Join the community discussions
