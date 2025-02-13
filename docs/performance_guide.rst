Performance Optimization Guide
==========================

This guide provides tips and best practices for optimizing Neural Circuit Policies using MLX, with a focus on Apple Silicon performance.

Apple Silicon Optimization
-----------------------

Hardware Features
~~~~~~~~~~~~~~~

MLX automatically leverages Apple Silicon features:

1. **Neural Engine**

.. code-block:: python

    # Enable Neural Engine optimizations
    import mlx.core as mx

    # MLX automatically uses the Neural Engine for supported operations
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            backbone_units=[64, 64]  # Sized for efficient Neural Engine usage
        )
    )

2. **Unified Memory**

.. code-block:: python

    # MLX efficiently manages unified memory
    # No explicit data transfers needed
    x = mx.random.normal((batch_size, seq_len, input_size))
    y = model(x)  # Data stays in unified memory

3. **Performance Cores**

.. code-block:: python

    # MLX automatically balances workload across cores
    @mx.compile(static_argnums=(1,))
    def process_batch(x, training=True):
        return model(x, training=training)

Device-Specific Tuning
~~~~~~~~~~~~~~~~~~~

Optimal configurations for different Apple Silicon chips:

.. code-block:: python

    def get_optimal_config(device_type):
        configs = {
            'M1': {
                'batch_size': 64,
                'backbone_units': [32, 32],
                'compile_static': True
            },
            'M1 Pro/Max': {
                'batch_size': 128,
                'backbone_units': [64, 64],
                'compile_static': True
            },
            'M1 Ultra': {
                'batch_size': 256,
                'backbone_units': [128, 128],
                'compile_static': True
            }
        }
        return configs.get(device_type, configs['M1'])

Memory Management
---------------

Lazy Evaluation
~~~~~~~~~~~~~

MLX's lazy evaluation system optimizes memory usage:

.. code-block:: python

    # Efficient computation graph
    def forward_pass(model, x, time_delta=None):
        # Operations are deferred
        outputs = model(x, time_delta=time_delta)
        
        # Compute only when needed
        if isinstance(outputs, tuple):
            return mx.eval(outputs[0]), mx.eval(outputs[1])
        return mx.eval(outputs)

Batch Processing
~~~~~~~~~~~~~~

Optimize batch sizes for your hardware:

.. code-block:: python

    class BatchOptimizer:
        def __init__(self, model):
            self.model = model
            
        def find_optimal_batch_size(self, start_size=32, max_size=512):
            sizes = []
            times = []
            
            for batch_size in [start_size * 2**i for i in range(5)]:
                if batch_size > max_size:
                    break
                    
                try:
                    x = mx.random.normal((batch_size, 100, self.model.input_size))
                    
                    # Warmup
                    _ = self.model(x)
                    mx.eval(_)
                    
                    # Timing
                    start = time.time()
                    for _ in range(10):
                        out = self.model(x)
                        mx.eval(out)
                    end = time.time()
                    
                    sizes.append(batch_size)
                    times.append((end - start) / 10)
                except:
                    break
                    
            return sizes[np.argmin(times)]

Memory-Efficient Training
~~~~~~~~~~~~~~~~~~~~~~

1. **Gradient Accumulation**

.. code-block:: python

    class GradientAccumulator:
        def __init__(self, model, optimizer, accum_steps=4):
            self.model = model
            self.optimizer = optimizer
            self.accum_steps = accum_steps
            
        def train_step(self, data_iterator):
            accumulated_grads = None
            total_loss = 0
            
            for i in range(self.accum_steps):
                x, y = next(data_iterator)
                loss, grads = self.compute_grads(x, y)
                total_loss += loss
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    for k, g in grads.items():
                        accumulated_grads[k] += g
            
            # Scale gradients
            for k in accumulated_grads:
                accumulated_grads[k] /= self.accum_steps
                
            self.optimizer.update(self.model, accumulated_grads)
            return total_loss / self.accum_steps

2. **Checkpointing**

.. code-block:: python

    class TrainingCheckpointer:
        def __init__(self, model, save_dir='checkpoints'):
            self.model = model
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
        def save(self, epoch, optimizer_state):
            state = {
                'model': self.model.state_dict(),
                'optimizer': optimizer_state,
                'epoch': epoch
            }
            path = f"{self.save_dir}/checkpoint_{epoch}.json"
            with open(path, 'w') as f:
                json.dump(state, f)
                
        def load(self, epoch):
            path = f"{self.save_dir}/checkpoint_{epoch}.json"
            with open(path, 'r') as f:
                state = json.load(f)
            self.model.load_state_dict(state['model'])
            return state['optimizer'], state['epoch']

Computation Optimization
---------------------

1. **MLX Compilation**

.. code-block:: python

    # Compile compute-intensive functions
    @mx.compile(static_argnums=(1, 2))
    def process_sequence(x, return_sequences=True, training=True):
        return model(x, return_sequences=return_sequences, training=training)

2. **Backbone Optimization**

.. code-block:: python

    # Efficient backbone configuration
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            backbone_units=[64, 64],  # Power of 2 for efficiency
            backbone_layers=2,
            backbone_dropout=0.1
        ),
        return_sequences=True
    )

3. **Time-Aware Processing**

.. code-block:: python

    class TimeOptimizer:
        def __init__(self, model):
            self.model = model
            
        @mx.compile(static_argnums=(1,))
        def process_batch(self, x, training=True):
            # Pre-compute time weights
            batch_size, seq_len = x.shape[:2]
            time_delta = mx.ones((batch_size, seq_len))
            
            # Process with time information
            return self.model(x, time_delta=time_delta, training=training)

Profiling and Monitoring
----------------------

1. **Memory Profiling**

.. code-block:: python

    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
            
        def take_snapshot(self):
            # Record memory usage
            snapshot = {
                'time': time.time(),
                'memory': mx.memory_stats()
            }
            self.snapshots.append(snapshot)
            
        def report(self):
            # Analyze memory usage patterns
            for snap in self.snapshots:
                print(f"Time: {snap['time']}, Memory: {snap['memory']}")

2. **Performance Monitoring**

.. code-block:: python

    class PerformanceMonitor:
        def __init__(self):
            self.metrics = defaultdict(list)
            
        def record(self, name, value):
            self.metrics[name].append(value)
            
        def report(self):
            for name, values in self.metrics.items():
                print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

Best Practices
------------

1. **Hardware Utilization**
   - Use power-of-2 sizes for tensors
   - Enable MLX compilation
   - Monitor memory usage
   - Profile performance

2. **Memory Management**
   - Leverage lazy evaluation
   - Use gradient accumulation
   - Implement checkpointing
   - Clear unused variables

3. **Computation**
   - Optimize backbone networks
   - Use time-aware processing
   - Implement efficient batching
   - Enable MLX optimizations

4. **Monitoring**
   - Profile memory usage
   - Monitor computation time
   - Track hardware utilization
   - Analyze bottlenecks

Common Issues
-----------

1. **Memory Issues**
   - Use smaller batch sizes
   - Implement gradient accumulation
   - Clear computation graphs
   - Monitor memory usage

2. **Performance Issues**
   - Enable MLX compilation
   - Optimize batch sizes
   - Use efficient architectures
   - Profile bottlenecks

3. **Training Issues**
   - Implement checkpointing
   - Monitor gradients
   - Track loss values
   - Validate results

Getting Help
----------

For performance-related issues:

1. Check example notebooks
2. Profile your code
3. Review this guide
4. File GitHub issues
5. Join discussions

References
---------

- `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
- `Apple Silicon Developer Guide <https://developer.apple.com/documentation/apple_silicon>`_
- `Neural Engine Documentation <https://developer.apple.com/documentation/coreml/core_ml_api/neural_engine>`_
- `Performance Best Practices <https://developer.apple.com/documentation/accelerate/performance_best_practices>`_
