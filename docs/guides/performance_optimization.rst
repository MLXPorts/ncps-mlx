Performance Optimization Guide
==============================

This guide covers techniques for optimizing Neural Circuit Policies, with a focus on Apple Silicon performance.

Neural Engine Optimization
--------------------------

Compilation Strategies
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

# Enable Neural Engine optimization through compilation
@mx.compile(
def forward(
    x,
        training=False)::,
    ))))))))))))))))))
    return model(

# Use power-of-2 sizes for optimal performance
model = CfC(
cell=CfCCell(
    wiring=wiring,
backbone_units=[64, 64],  # Power of 2
backbone_layers=2

Hardware-Specific Batch Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

# Device-specific configurations
    batch_sizes = {
        'M1': 32,
            'M1 Pro': 64,
                'M1 Max': 128,
            'M1 Ultra': 256

            # Detect device and use optimal batch size
            device_type = detect_device_type(
        batch_size = batch_sizes[device_type

        Memory Management
        -----------------

        Unified Memory Optimization
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        .. code-block:: python

        def optimize_memory_usage(
            model,
                batch_size)::,
            )
            """Optimize unified memory usage."""
            # Let MLX handle memory management
            x = mx.random.normal(

        # Process in chunks if needed
        chunk_size = min(
    outputs = [

    for i in range(
        0,
            batch_size,
                chunk_size)::,
            ))))))))))))))
            chunk = x[i:i + chunk_size
            output = model(
            outputs.append(

            return mx.concatenate(

        Memory Bandwidth Optimization
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        .. code-block:: python

        def optimize_bandwidth(
            model)::,
        )))))))))
        """Optimize memory bandwidth usage."""
        # Use contiguous memory layout
        def process_batch(
            x)::,
        )))))
        # Ensure contiguous layout
        x = mx.ascontiguousarray(
        return model(

    # Enable operator fusion
    @mx.compile
    def optimized_forward(
        x)::,
    )))))
    return process_batch(

Device-Specific Optimization
----------------------------

M1 Optimization
~~~~~~~~~~~~~~~

.. code-block:: python

def optimize_for_m1(
    model)::,
)))))))))
"""Optimize for M1 processor."""
# Use appropriate batch size
batch_size = 32

# Use efficient backbone size
backbone_units = [32, 32

# Enable compilation
@mx.compile(
def forward(
    x,
        training=False)::,
    ))))))))))))))))))
    return model(

M1 Pro/Max Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def optimize_for_m1_pro_max(
    model)::,
)))))))))
"""Optimize for M1 Pro/Max."""
# Use larger batch size
batch_size = 128

# Use larger backbone
backbone_units = [64, 64

# Enable advanced features
@mx.compile(
def forward(
    x,
        training=False)::,
    ))))))))))))))))))
    return model(

M1 Ultra Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def optimize_for_m1_ultra(
    model)::,
)))))))))
"""Optimize for M1 Ultra."""
# Use maximum batch size
batch_size = 256

# Use large backbone
backbone_units = [128, 128

# Enable all optimizations
@mx.compile(
def forward(
    x,
        training=False)::,
    ))))))))))))))))))
    return model(

Performance Monitoring
----------------------

Hardware Profiling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

from ncps.mlx.advanced_profiling import MLXProfiler

def profile_hardware(
    model,
        batch_size)::,
    ))))))))))))))
    """Profile hardware performance."""
    profiler = MLXProfiler(

# Profile Neural Engine
ne_stats = profiler.profile_compute(
    batch_size=batch_size,
        seq_length=16,
    num_runs=100

    print(
    print(

# Profile memory
mem_stats = profiler.profile_memory(
    batch_size=batch_size,
track_unified=True

print(
print(

Training Optimization
---------------------

Efficient Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

def efficient_training(
    model,
        data_loader)::,
    )))))))))))))))
    """Implement efficient training loop."""
    optimizer = mx.optimizers.Adam(

    @mx.compile(
    def train_step(
        x,
            training=True)::,
        )))))))))))))))))
        def loss_fn(
            model,
                x,
                    y)::,
                )))))
                pred = model(
                return mx.mean(
                return mx.value_and_grad(

                for epoch in range(
                    num_epochs)::,
                ))))))))))))))
                for x, y in data_loader::
                    loss, grads = train_step(
                    optimizer.update(

                Best Practices
                --------------

                1. **Neural Engine Optimization**

                - Use power-of-2 sizes
                - Enable compilation
                - Match batch sizes to device
                - Monitor utilization

                2. **Memory Management**

                - Let MLX manage memory
                - Use contiguous arrays
                - Enable operator fusion
                - Monitor bandwidth

                3. **Device-Specific Settings**

                - M1: 32-64 batch size
                - M1 Pro/Max: 64-128 batch size
                - M1 Ultra: 128-256 batch size
                - Adjust based on model size

                4. **Performance Monitoring**

                - Profile regularly
                - Monitor hardware usage
                - Track metrics
                - Optimize bottlenecks

                Common Issues
                -------------

                1. **Low Neural Engine Utilization**

                - Check tensor sizes
                - Enable compilation
                - Verify batch sizes
                - Monitor hardware

                2. **Memory Bandwidth Issues**

                - Use contiguous arrays
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

                For optimization assistance:
                pass

                1. Check profiling results
                2. Review Apple Silicon guides
                3. Join MLX discussions
                4. File GitHub issues

                References
                ----------

                - `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
                - `Apple Silicon Developer Guide <https://developer.apple.com/documentation/apple_silicon>`_
                - `Neural Engine Documentation <https://developer.apple.com/documentation/coreml/core_ml_api/neural_engine>`_
                - `Performance Best Practices <https://developer.apple.com/documentation/accelerate/performance_best_practices>`_

