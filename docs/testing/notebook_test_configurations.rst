Notebook Test Configurations
============================

This document outlines test configurations and validation strategies for
MLX notebooks running on Apple Silicon.

Test Environment Setup
----------------------

1. Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- Minimum: M1 with 8GB RAM
- Recommended: M1 Pro/Max with 16GB+ RAM
- Optimal: M1 Ultra with 64GB+ RAM

2. Software Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

# Core dependencies
pip install mlx
pip install pytest pytest-notebook nbval

# Development dependencies
pip install jupyter nbconvert nbformat

3. Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

# Enable Neural Engine
export MLX_USE_NEURAL_ENGINE=1

# Enable debug logging
export MLX_DEBUG_LOG=1

# Set device type (for testing)
export DEVICE_TYPE="M1"  # or "M1 Pro", "M1 Max", "M1 Ultra"

Validation Framework
--------------------

1. Performance Tests
~~~~~~~~~~~~~~~~~~~~

.. code:: python

class NotebookPerformanceTests:
    """Performance validation for notebooks."""

    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.config = get_device_config()

    def validate_compute_performance(self):
        """Validate compute performance."""
        stats = profile_notebook_compute()
        assert stats['tflops'] >= self.config.min_tflops

    def validate_memory_usage(self):
        """Validate memory usage."""
        stats = profile_notebook_memory()
        assert stats['peak_memory'] <= self.config.memory_budget

    def validate_bandwidth(self):
        """Validate memory bandwidth."""
        stats = profile_notebook_bandwidth()
        assert stats['bandwidth'] >= self.config.min_bandwidth

2. Code Quality Tests
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class NotebookQualityTests:
    """Code quality validation for notebooks."""

    def validate_imports(self):
        """Validate import organization."""
        required_imports = [
            'mlx.core',
            'mlx.nn',
            'mlx.optimizers',
            'ncps.mlx',
            'ncps.wirings'
        ]
        # Implementation

    def validate_structure(self):
        """Validate notebook structure."""
        required_sections = [
            'Setup',
            'Model Creation',
            'Training',
            'Evaluation'
        ]
        # Implementation

    def validate_documentation(self):
        """Validate documentation completeness."""
        required_docs = [
            'Hardware Requirements',
            'Performance Notes',
            'Usage Instructions'
        ]
        # Implementation

3. Hardware-Specific Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

class HardwareSpecificTests:
    """Hardware-specific validation."""

    def validate_neural_engine(self):
        """Validate Neural Engine usage."""
        stats = profile_neural_engine()
        assert stats['utilization'] >= 50  # >50% utilization

    def validate_batch_sizes(self):
        """Validate batch size selection."""
        batch_size = get_notebook_batch_size()
        assert batch_size in self.config.batch_sizes

    def validate_model_sizes(self):
        """Validate model size selection."""
        hidden_size = get_notebook_hidden_size()
        assert hidden_size in self.config.hidden_sizes

Test Configurations
-------------------

1. Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

BASIC_CONFIG = {
    'skip_performance': False,
    'skip_hardware': False,
    'skip_quality': False,
    'execution_timeout': 600,  # seconds
    'memory_limit': 8192,  # MB
    'min_tflops': 1.0,
'min_bandwidth': 50.0  # GB/s
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

2. CI/CD Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

CICD_CONFIG = {
    'skip_performance': True,  # Skip in CI
    'skip_hardware': False,
    'skip_quality': False,
    'execution_timeout': 300,
    'memory_limit': 4096,
    'min_tflops': 0.5,  # Lower requirements for CI
'min_bandwidth': 25.0
}}}}}}}}}}}}}}}}}}}}}

3. Development Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

DEV_CONFIG = {
    'skip_performance': False,
    'skip_hardware': False,
    'skip_quality': False,
    'execution_timeout': 1200,
    'memory_limit': 16384,
    'min_tflops': 2.0,
'min_bandwidth': 100.0
}}}}}}}}}}}}}}}}}}}}}}

Test Execution
--------------

1. Running Tests
~~~~~~~~~~~~~~~~

.. code:: bash

# Run all tests
pytest --notebook-test-mode=all notebooks/

# Run specific tests
pytest --notebook-test-mode=performance notebooks/mlx_benchmarks.ipynb
pytest --notebook-test-mode=quality notebooks/mlx_cfc_example.ipynb

2. Test Reports
~~~~~~~~~~~~~~~

.. code:: python

def generate_test_report(results):
    """Generate test report."""
    report = {
        'performance': {
            'tflops': results['tflops'],
            'bandwidth': results['bandwidth'],
            'memory': results['memory']
        },
        'quality': {
            'imports': results['imports_valid'],
            'structure': results['structure_valid'],
            'documentation': results['docs_valid']
        },
        'hardware': {
            'neural_engine': results['ne_utilization'],
            'batch_size': results['batch_size_valid'],
            'model_size': results['model_size_valid']
        }
    }
    return report

Validation Criteria
-------------------

1. Performance Criteria
~~~~~~~~~~~~~~~~~~~~~~~

- TFLOPS meets device minimum
- Memory usage within budget
- Bandwidth meets requirements
- Neural Engine properly utilized

2. Code Quality Criteria
~~~~~~~~~~~~~~~~~~~~~~~~

- Proper import organization
- Clear notebook structure
- Complete documentation
- Hardware-specific notes

3. Hardware Utilization Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Neural Engine enabled
- Appropriate batch sizes
- Optimal model sizes
- Efficient memory usage

Best Practices
--------------

1. Test Development
~~~~~~~~~~~~~~~~~~~

- Write comprehensive tests
- Include all validation types
- Test on multiple devices
- Document test requirements

2. Test Maintenance
~~~~~~~~~~~~~~~~~~~

- Update test configurations
- Monitor performance metrics
- Track hardware changes
- Update documentation

.. _test-execution-1:

3. Test Execution
~~~~~~~~~~~~~~~~~

- Run tests regularly
- Monitor test results
- Update test criteria
- Document failures

Resources
---------

1. MLX Testing Guide
2. Apple Silicon Testing Guide
3. Performance Testing Guide
4. Notebook Testing Tools
