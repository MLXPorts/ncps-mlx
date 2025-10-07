# NCPSLint - Code Quality Tool for ncps-mlx

NCPSLint is a customized version of EmberLint tailored specifically for the ncps-mlx codebase. It helps maintain code quality by checking for common issues and enforcing MLX-specific best practices.

## What It Checks

### ✓ Always Checked
- **Syntax Errors**: Python syntax issues
- **Import Errors**: Invalid or missing imports
- **Compilation Errors**: Issues that prevent code from running

### ⚠️ Code Quality Issues
- **NumPy Usage**: Detects NumPy imports and usage (prefer MLX)
- **Unused Imports**: Identifies imports that aren't used
- **Precision-Reducing Casts**: Warns about `float()` and `int()` casts that may lose precision
- **Python Operators**: Tracks usage of `+`, `-`, `*`, `/` operators (normal in MLX)

### ❌ Disabled for ncps-mlx
- **Tensor Conversions**: Not applicable (MLX-only codebase)
- **Backend Consistency**: Not applicable (single backend)
- **Frontend-Backend Separation**: Not applicable (MLX-specific)

## Usage

### Basic Scan
```bash
# Scan the entire ncps/mlx directory
python3 misc/emberlint.py ncps/mlx

# Scan with detailed output
python3 misc/emberlint.py ncps/mlx --verbose
```

### Focused Checks
```bash
# Check for NumPy usage (most important for MLX purity)
python3 misc/emberlint.py ncps/mlx --numpy-only --verbose

# Check for unused imports (easy to fix)
python3 misc/emberlint.py ncps/mlx --unused-only --verbose

# Check for precision-reducing casts
python3 misc/emberlint.py ncps/mlx --precision-only --verbose

# Check only syntax
python3 misc/emberlint.py ncps/mlx --syntax-only
```

### Exclude Directories
```bash
# Skip tests and examples
python3 misc/emberlint.py ncps/mlx --exclude tests examples
```

## Interpreting Results

### Current Status (as of cleanup)
```
Total files analyzed: 44
Files with syntax errors: 0 (0.00%)        ✓ Clean
Files with NumPy: 7 (15.91%)               ⚠️ Should review
Files with precision casts: 6 (13.64%)     ⚠️ Should review
Files with tensor conversions: 0 (0.00%)   ✓ Clean
Files with Python operators: 30 (68.18%)   ✓ Normal for MLX
Files with unused imports: 32 (72.73%)     ⚠️ Can clean up
Files with backend-specific code: 0        ✓ Clean (was 40 before fix)
```

### Priority Actions
1. **NumPy Usage (7 files)**: Review if MLX can be used instead
   - Some NumPy usage is acceptable for preprocessing/testing
   - Core computation should use MLX

2. **Unused Imports (32 files)**: Clean up to improve code clarity
   - Run with `--unused-only --verbose` to see specific imports
   - Easy wins for code quality

3. **Precision Casts (6 files)**: Verify they don't cause issues
   - Check if `float()` or `int()` casts are necessary
   - Consider using MLX's casting functions

4. **Type Errors (44 files)**: Can be ignored
   - These are false positives from MLX's type stub file
   - Not actual code issues

## Best Practices

### When to Use MLX vs NumPy
✓ **Use MLX for:**
- Tensor operations
- Neural network computations
- Gradient calculations
- GPU-accelerated operations

✓ **NumPy is OK for:**
- Data loading/preprocessing
- Test assertions
- Simple array creation before conversion to MLX
- Visualization data preparation

### Fixing Common Issues

#### Unused Imports
```python
# Before
import mlx.core as mx
import mlx.nn as nn
from typing import Optional  # unused

# After
import mlx.core as mx
import mlx.nn as nn
```

#### NumPy Usage
```python
# Before
import numpy as np
x = np.array([1, 2, 3])

# After
import mlx.core as mx
x = mx.array([1, 2, 3])
```

#### Precision Casts
```python
# Before
value = float(tensor)  # May lose precision

# After
value = mx.astype(tensor, mx.float32)  # Explicit type
```

## Integration

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python3 misc/emberlint.py ncps/mlx --numpy-only
if [ $? -ne 0 ]; then
    echo "NCPSLint found NumPy usage. Consider using MLX instead."
fi
```

### CI/CD
Add to your CI pipeline:
```yaml
- name: Run NCPSLint
  run: |
    python3 misc/emberlint.py ncps/mlx
    python3 misc/emberlint.py ncps/mlx --numpy-only --verbose
```

## Customization

The tool is configured specifically for ncps-mlx but can be adjusted by modifying:
- `ALLOW_SINGLE_ISSUE_LINTING`: Enable/disable single-issue mode
- Backend checks: Disabled for MLX-only codebase
- Tensor conversion checks: Disabled for MLX

## Questions?

- The tool is adapted from EmberLint for multi-backend ML frameworks
- Customized to focus on MLX purity and code quality
- Backend-agnostic features have been disabled or adapted
