# MLX Python Operator Fixes Required

## Critical Issue

Python operators (`+`, `-`, `*`, `/`, `**`) on MLX arrays **break the computation graph** by forcing materialization to CPU. This defeats MLX's lazy evaluation and GPU acceleration.

## ✅ Fixed Files (3)

- `ncps/mlx/cfc_cell.py` - Core Closed-form Continuous-time cell
- `ncps/mlx/ltc4_cell.py` - Liquid Time-Constant variant  
- `ncps/mlx/icra_cfc_cell.py` - LIDAR-specific CfC for robot navigation

## ⚠️ Files Still Needing Fixes (12)

Ranked by number of violations and priority:

### High Priority - Core Cell Implementations
1. **ltcse_cell.py** - 67 violations - Liquid Time-Constant with Squeeze-Excitation
2. **ltc_cell.py** - 27 violations - Core LTC implementation
3. **ode_solvers.py** - 29 violations - ODE integration utilities (shared by multiple cells)

### Medium Priority - Advanced Cells
4. **ctgru_se_cell.py** - 19 violations - Continuous-Time GRU with SE
5. **node_se_cell.py** - 17 violations - Neural ODE with SE
6. **eltc_cell.py** - 6 violations - Enhanced LTC
7. **ctrnn_se_cell.py** - 5 violations - Continuous-Time RNN with SE

### Low Priority - Utilities & Wrappers
8. **module_training_demo.py** - 4 violations - Training example
9. **liquid_utils.py** - 4 violations - Utility functions
10. **wired_cfc_cell.py** - 2 violations - Wired CfC variant
11. **hyperprofiles.py** - 1 violation - Hyperparameter profiles
12. **ctgru.py** - 1 violation - CT-GRU wrapper

## How to Fix

Replace Python operators with MLX functions:

```python
# ❌ WRONG - Breaks computation graph
result = a + b * c - d / e

# ✅ CORRECT - Preserves computation graph  
result = mx.subtract(mx.add(a, mx.multiply(b, c)), mx.divide(d, e))

# Or more readable with intermediate variables:
temp1 = mx.multiply(b, c)
temp2 = mx.add(a, temp1)
temp3 = mx.divide(d, e)
result = mx.subtract(temp2, temp3)
```

### Operator Mapping

| Python Op | MLX Function |
|-----------|--------------|
| `a + b` | `mx.add(a, b)` |
| `a - b` | `mx.subtract(a, b)` |
| `a * b` | `mx.multiply(a, b)` |
| `a / b` | `mx.divide(a, b)` |
| `a // b` | `mx.floor_divide(a, b)` |
| `a % b` | `mx.remainder(a, b)` |
| `a ** b` | `mx.power(a, b)` |
| `-a` | `mx.negative(a)` |
| `a @ b` | `mx.matmul(a, b)` (already used correctly) |

### Important Exceptions

**Python operators ARE allowed for:**
- Array indexing: `x[i + 1]`, `x[start:end+size]` ✅
- Pure Python integers/floats: `for i in range(n-1)` ✅  
- String operations: `path = base + ".json"` ✅

**Only fix operators that operate on `mx.array` objects!**

## Verification

Run emberlint to check progress:

```bash
# Check all files
python ./misc/emberlint.py ncps/mlx/ --verbose

# Check specific file
python ./misc/emberlint.py ncps/mlx/ltcse_cell.py --verbose

# Count remaining violations
python ./misc/emberlint.py ncps/mlx/ --verbose 2>&1 | grep "Files with Python operators:"
```

## Why This Matters

From Metal/MLX architecture perspective:

1. **Lazy Evaluation**: MLX builds a computation graph that executes on GPU
2. **Python Operators**: Force immediate evaluation, breaking the graph
3. **Performance**: Each break causes a GPU→CPU sync, destroying parallelism
4. **Memory**: Prevents fusion optimizations, increases memory traffic

Think like a C/Metal programmer: keep buffers on GPU, chain operations in the graph, only materialize when absolutely necessary (like printing or branching on values).

## References

- `/Volumes/emberstuff/xLSTM/docs/metal_reference/` - MLX/Metal best practices
- C. elegans tap-withdrawal paper (ICRA 2018) - Neuronal Circuit Policies
- emberlint.py - Linting tool that catches these violations
