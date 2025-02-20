# Sphinx Conversion Example

This document demonstrates how our TensorAbstraction documentation would look when converted to Sphinx RST format.

## Current Markdown Format
```markdown
# TensorAbstraction

## Overview
TensorAbstraction provides a unified interface for tensor operations across different backends, with automatic backend selection based on hardware availability and configurable precedence.

## Core Features
1. Backend Auto-Detection
2. Tensor Operations
3. Automatic Conversion

## Usage Examples
```python
# Uses best available backend
x = TensorAbstraction.tensor([[1, 2], [3, 4]])
y = TensorAbstraction.matmul(x, x)
```
```

## Equivalent RST Format
```rst
TensorAbstraction
================

Overview
--------
TensorAbstraction provides a unified interface for tensor operations across different 
backends, with automatic backend selection based on hardware availability and 
configurable precedence.

Core Features
------------
.. contents::
   :local:
   :depth: 1

Backend Auto-Detection
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python
   :caption: Backend Selection
   :emphasize-lines: 2

   # Get optimal backend
   backend = TensorAbstraction.get_optimal_backend()

Tensor Operations
~~~~~~~~~~~~~~~
.. code-block:: python
   :caption: Basic Operations

   x = TensorAbstraction.tensor([[1, 2], [3, 4]])
   y = TensorAbstraction.matmul(x, x)

.. note::
   Operations automatically use the optimal backend for the current hardware.

Implementation Details
--------------------
.. autoclass:: ncps.abstractions.TensorAbstraction
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------
.. toctree::
   :maxdepth: 1

   layer
   gpu
   liquid

.. seealso::
   - :doc:`/architecture/implementation/phase1`
   - :doc:`/guides/performance_optimization`
```

## Key Sphinx Features Used

1. Auto-Generated Table of Contents
   ```rst
   .. contents::
      :local:
      :depth: 1
   ```

2. Code Blocks with Highlighting
   ```rst
   .. code-block:: python
      :caption: Example Code
      :emphasize-lines: 2,3
   ```

3. Cross-References
   ```rst
   :doc:`/architecture/implementation/phase1`
   ```

4. API Documentation
   ```rst
   .. autoclass:: ncps.abstractions.TensorAbstraction
      :members:
   ```

5. Notes and Warnings
   ```rst
   .. note::
      Important information here.

   .. warning::
      Critical warning here.
   ```

6. See Also Links
   ```rst
   .. seealso::
      Related documentation links.
   ```

## Benefits of RST Format

1. Better Integration
   - Automatic API documentation
   - Cross-references to other docs
   - Integration with Python docstrings

2. Enhanced Features
   - Code syntax highlighting
   - Line emphasis
   - Automatic TOC generation

3. Multiple Output Formats
   - HTML documentation
   - PDF generation
   - ePub format

4. Development Benefits
   - Consistent formatting
   - Better navigation
   - Searchable documentation

This example demonstrates how our existing markdown documentation can be enhanced using Sphinx's RST features while maintaining clarity and organization.