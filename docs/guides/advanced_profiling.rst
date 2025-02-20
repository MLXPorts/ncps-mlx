Advanced Profiling Guide
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
========================
====================

This guide covers advanced profiling tools for Neural Circuit Policies with a focus on Apple Silicon optimization using MLX.

Apple Silicon Profiling
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-------------------

Overview
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~

MLX provides specialized profiling tools for Apple Silicon processors:

- Neural Engine utilization
- Unified memory analysis
- Hardware-specific optimizations
- Performance counters

Basic Usage
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
-----------
--------

Quick Profile
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
~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.advanced_profiling import quick_profile
    
    # Create model with Neural Engine-friendly configuration
    wiring = AutoNCP(
        units=64,  # Power of 2 size
        output_size=16
    )
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            backbone_units=[64, 64],  # Power of 2 sizes
            backbone_layers=2
        )
    )
    
    # Quick profile with compilation
    @mx.compile(static_argnums=(1,))
    def forward(x, training=False):
        return model(x, training=training)
    
    stats = quick_profile(
        model,
        batch_size=32,
        seq_length=16,  # Power of 2
        num_runs=100,
        forward_fn=forward
    )

Neural Engine Profiling
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
------------------

Compute Profiling
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
~~~~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.advanced_profiling import MLXProfiler
    
    profiler = MLXProfiler(model)
    
    # Profile Neural Engine computation
    stats = profiler.profile_compute(
        batch_size=32,
        seq_length=16,
        num_runs=100,
        enable_ne=True  # Enable Neural Engine
    )
    
    print(f"Neural Engine TFLOPS: {stats['ne_tflops']:.2f}")
    print(f"Utilization: {stats['ne_utilization']:.2f}%")
    print(f"Average time: {stats['time_mean']*1000:.2f} ms")

Memory Profiling
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
~~~~~~~~~~~~

.. code-block:: python

    # Profile unified memory usage
    stats = profiler.profile_memory(
        batch_size=32,
        seq_length=16,
        track_unified=True  # Track unified memory
    )
    
    print(f"Peak unified memory: {stats['unified_peak']:.2f} MB")
    print(f"Bandwidth: {stats['bandwidth']:.2f} GB/s")
    print(f"Total allocated: {stats['total_allocated']:.2f} MB")

Hardware Counters
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
~~~~~~~~~~~~~

.. code-block:: python

    # Profile hardware counters
    stats = profiler.profile_hardware(
        batch_size=32,
        seq_length=16
    )
    
    print(f"Neural Engine cycles: {stats['ne_cycles']}")
    print(f"Memory bandwidth: {stats['memory_bandwidth']} GB/s")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}%")

Performance Analysis
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
----------------

Device-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze performance across Apple Silicon devices
    devices = ['M1', 'M1 Pro', 'M1 Max', 'M1 Ultra']
    batch_sizes = {
        'M1': 32,
        'M1 Pro': 64,
        'M1 Max': 128,
        'M1 Ultra': 256
    }
    
    results = []
    for device in devices:
        stats = profiler.profile_compute(
            batch_size=batch_sizes[device],
            seq_length=16,
            device=device
        )
        results.append({
            'device': device,
            'tflops': stats['tflops'],
            'memory': stats['peak_memory']
        })

Memory Bandwidth Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze memory bandwidth utilization
    def analyze_bandwidth(batch_sizes=[32, 64, 128]):
        results = []
        for batch_size in batch_sizes:
            stats = profiler.profile_memory(
                batch_size=batch_size,
                seq_length=16,
                track_bandwidth=True
            )
            results.append({
                'batch_size': batch_size,
                'bandwidth': stats['bandwidth'],
                'utilization': stats['bandwidth_utilization']
            })
        return results

Neural Engine Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Optimize for Neural Engine
    def optimize_for_ne(model, input_shape):
        # Enable Neural Engine profiling
        profiler = MLXProfiler(model)
        
        # Test different configurations
        configs = [
            {'compile': False, 'ne': False},
            {'compile': True, 'ne': False},
            {'compile': True, 'ne': True}
        ]
        
        results = []
        for config in configs:
            if config['compile']:
                @mx.compile(static_argnums=(1,))
                def forward(x, training=False):
                    return model(x, training=training)
            else:
                forward = lambda x, training: model(x, training=training)
            
            stats = profiler.profile_compute(
                input_shape=input_shape,
                forward_fn=forward,
                enable_ne=config['ne']
            )
            
            results.append({
                'config': config,
                'tflops': stats['tflops'],
                'time': stats['time_mean']
            })
        
        return results

Visualization
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
----------

Performance Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Plot Neural Engine performance
    def plot_ne_performance(results):
        plt.figure(figsize=(12, 4))
        
        # Plot TFLOPS
        plt.subplot(131)
        plt.bar(range(len(results)),
                [r['tflops'] for r in results],
                tick_label=[r['device'] for r in results])
        plt.ylabel('TFLOPS')
        plt.title('Neural Engine Performance')
        
        # Plot memory bandwidth
        plt.subplot(132)
        plt.bar(range(len(results)),
                [r['bandwidth'] for r in results],
                tick_label=[r['device'] for r in results])
        plt.ylabel('GB/s')
        plt.title('Memory Bandwidth')
        
        # Plot utilization
        plt.subplot(133)
        plt.bar(range(len(results)),
                [r['utilization'] for r in results],
                tick_label=[r['device'] for r in results])
        plt.ylabel('Utilization %')
        plt.title('Hardware Utilization')
        
        plt.tight_layout()
        plt.show()

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
-----------

1. **Neural Engine Optimization**

   - Use power-of-2 sizes
   - Enable compilation
   - Monitor utilization
   - Profile different configs

2. **Memory Management**

   - Track unified memory
   - Monitor bandwidth
   - Optimize transfers
   - Profile allocations

3. **Hardware Utilization**

   - Match batch sizes to device
   - Monitor counters
   - Optimize for hardware
   - Track performance

4. **Performance Tuning**

   - Profile regularly
   - Test configurations
   - Monitor metrics
   - Optimize bottlenecks

Device-Specific Settings
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
------------------------
--------------------

1. **M1**

   - Batch size: 32-64
   - Memory budget: ~8GB
   - Neural Engine: Enable
   - Compilation: Required

2. **M1 Pro/Max**

   - Batch size: 64-128
   - Memory budget: ~16-32GB
   - Neural Engine: Enable
   - Compilation: Required

3. **M1 Ultra**

   - Batch size: 128-256
   - Memory budget: ~64GB
   - Neural Engine: Enable
   - Compilation: Required

Troubleshooting
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
------------

Common Issues
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
~~~~~~~~~~

1. **Low Neural Engine Utilization**

   - Check tensor sizes
   - Enable compilation
   - Verify configurations
   - Monitor counters

2. **Memory Bandwidth Issues**

   - Check transfer patterns
   - Optimize batch sizes
   - Monitor unified memory
   - Profile bandwidth

3. **Performance Problems**

   - Profile bottlenecks
   - Check configurations
   - Monitor utilization
   - Optimize patterns

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

For profiling assistance:

1. Check profiling results
2. Review Apple Silicon guides
3. Join MLX discussions
4. File GitHub issues
