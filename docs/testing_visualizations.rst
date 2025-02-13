Testing Visualization Extensions
==========================

This guide covers best practices for testing visualization extensions, with special focus on Apple Silicon optimizations.

Apple Silicon Testing
----------------

1. **Neural Engine Tests**
   
   Test Neural Engine utilization:

   .. code-block:: python

       import unittest
       import mlx.core as mx
       from ncps.mlx.visualization import HardwareVisualizer
       
       class TestNeuralEngine(unittest.TestCase):
           def setUp(self):
               self.visualizer = HardwareVisualizer(model)
           
           def test_ne_utilization(self):
               # Test Neural Engine utilization visualization
               stats = self.visualizer.profile_ne()
               self.assertIsNotNone(stats['ne_utilization'])
               self.assertGreaterEqual(stats['ne_utilization'], 0)
               self.assertLessEqual(stats['ne_utilization'], 100)
           
           def test_compilation_effect(self):
               # Test effect of compilation
               uncompiled_stats = self.visualizer.profile_uncompiled()
               
               @mx.compile(static_argnums=(1,))
               def forward(x, training=False):
                   return self.model(x, training=training)
               
               compiled_stats = self.visualizer.profile_compiled(forward)
               self.assertGreater(
                   compiled_stats['tflops'],
                   uncompiled_stats['tflops']
               )

2. **Memory Tests**
   
   Test unified memory visualization:

   .. code-block:: python

       class TestMemoryVisualization(unittest.TestCase):
           def test_memory_tracking(self):
               # Test memory usage visualization
               memory_viz = self.visualizer.visualize_memory_usage()
               self.assertIsNotNone(memory_viz['bandwidth'])
               self.assertIsNotNone(memory_viz['utilization'])
           
           def test_bandwidth_monitoring(self):
               # Test bandwidth visualization
               bandwidth_data = self.visualizer.monitor_bandwidth()
               self.assertGreater(len(bandwidth_data), 0)
               self.assertGreater(bandwidth_data['peak_bandwidth'], 0)

3. **Hardware-Specific Tests**
   
   Test device-specific features:

   .. code-block:: python

       class TestHardwareSpecific(unittest.TestCase):
           def test_device_optimization(self):
               # Test device-specific visualizations
               device_stats = self.visualizer.profile_device()
               self.assertIn('device_type', device_stats)
               self.assertIn('optimal_batch_size', device_stats)
           
           def test_performance_scaling(self):
               # Test performance scaling visualization
               scaling_data = self.visualizer.visualize_scaling()
               self.assertGreater(len(scaling_data['batch_sizes']), 0)
               self.assertGreater(len(scaling_data['tflops']), 0)

Unit Testing
---------

[Previous unit testing section remains the same...]

Integration Testing
---------------

[Previous integration testing section remains the same...]

Performance Testing
---------------

1. **Hardware-Aware Memory Testing**
   
   Test memory efficiency with hardware considerations:

   .. code-block:: python

       import memory_profiler
       
       class TestHardwarePerformance(unittest.TestCase):
           @profile
           def test_memory_usage(self):
               # Test memory usage during visualization
               initial_mem = memory_profiler.memory_usage()[0]
               self.visualizer.create_visualization()
               final_mem = memory_profiler.memory_usage()[0]
               
               # Check memory increase is reasonable for device
               device_memory = self.get_device_memory()
               max_usage = device_memory * 0.1  # 10% of device memory
               self.assertLess(final_mem - initial_mem, max_usage)

2. **Neural Engine Performance**
   
   Test Neural Engine utilization:

   .. code-block:: python

       class TestNeuralEnginePerformance(unittest.TestCase):
           def test_ne_efficiency(self):
               # Test Neural Engine efficiency
               stats = self.visualizer.profile_ne_performance()
               self.assertGreater(stats['ne_utilization'], 50)  # >50% utilization
               self.assertGreater(stats['tflops'], 1.0)  # >1 TFLOPS

3. **Device-Specific Scaling**
   
   Test performance scaling on different devices:

   .. code-block:: python

       class TestDeviceScaling(unittest.TestCase):
           def test_batch_size_scaling(self):
               # Test scaling with device-specific batch sizes
               device_type = self.get_device_type()
               batch_sizes = {
                   'M1': [32, 64],
                   'M1 Pro': [64, 128],
                   'M1 Max': [128, 256],
                   'M1 Ultra': [256, 512]
               }
               
               for batch_size in batch_sizes[device_type]:
                   perf = self.visualizer.profile_batch_size(batch_size)
                   self.assertGreater(perf['efficiency'], 0.7)  # >70% efficient

Visual Testing
-----------

[Previous visual testing section remains the same...]

Continuous Integration
------------------

1. **Hardware-Specific CI**
   
   Setup device-specific testing:

   .. code-block:: yaml

       # .github/workflows/test-visualizations.yml
       name: Test Visualizations
       
       on: [push, pull_request]
       
       jobs:
         test-apple-silicon:
           runs-on: self-hosted
           strategy:
             matrix:
               device: ['M1', 'M1 Pro', 'M1 Max', 'M1 Ultra']
           steps:
           - uses: actions/checkout@v2
           - name: Set up Python
             uses: actions/setup-python@v2
           - name: Install dependencies
             run: |
               pip install -r requirements.txt
               pip install pytest pytest-cov
           - name: Run tests
             run: |
               DEVICE_TYPE=${{ matrix.device }} pytest tests/visualization

Best Practices
-----------

1. **Hardware-Aware Testing**
   - Test on all target devices
   - Monitor hardware utilization
   - Profile performance metrics
   - Validate optimization effects

2. **Memory Management**
   - Test unified memory usage
   - Monitor bandwidth utilization
   - Profile cache performance
   - Validate memory patterns

3. **Performance Optimization**
   - Test compilation effects
   - Validate batch sizes
   - Monitor Neural Engine
   - Profile hardware usage

4. **Error Handling**
   - Test hardware-specific errors
   - Validate error recovery
   - Monitor resource usage
   - Log hardware states

Getting Started
------------

[Previous getting started section remains the same...]

References
--------

- `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
- `Apple Silicon Developer Guide <https://developer.apple.com/documentation/apple_silicon>`_
- `Neural Engine Documentation <https://developer.apple.com/documentation/coreml/core_ml_api/neural_engine>`_
- `Performance Best Practices <https://developer.apple.com/documentation/accelerate/performance_best_practices>`_
