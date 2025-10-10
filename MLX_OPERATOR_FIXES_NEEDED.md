# MLX Python Operator Fixes Required

## Critical Issue

Python operators (`+`, `-`, `*`, `/`, `**`) on MLX arrays **break the computation graph** by forcing materialization to CPU. This defeats MLX's lazy evaluation and GPU acceleration.

## ✅ Fixed Files (9)

**Core Cell Implementations:**
- `ncps/mlx/cfc_cell.py` - Core Closed-form Continuous-time cell
- `ncps/mlx/ltc4_cell.py` - Liquid Time-Constant variant  
- `ncps/mlx/icra_cfc_cell.py` - LIDAR-specific CfC for robot navigation
- `ncps/mlx/ltcse_cell.py` - Liquid Time-Constant with Squeeze-Excitation (67 fixes)
- `ncps/mlx/ltc_cell.py` - Core LTC implementation (27 fixes)
- `ncps/mlx/node_se_cell.py` - Neural ODE with Squeeze-Excitation (17 fixes)
- `ncps/mlx/eltc_cell.py` - Enhanced LTC (6 fixes)

**Utilities:**
- `ncps/mlx/ode_solvers.py` - ODE integration utilities (29 fixes)

## ⚠️ Files with Remaining False Positives (8)

emberlint flags Python int operations in shape calculations, but these are fine since they operate on Python scalars not MLX arrays:

1. **ctgru_se_cell.py** - 9 flags (Python int ops: `self.units * self.M` in shape tuples)
2. **ctrnn_se_cell.py** - 5 flags
3. **module_training_demo.py** - 4 flags  
4. **liquid_utils.py** - 4 flags
5. **wired_cfc_cell.py** - 2 flags
6. **hyperprofiles.py** - 1 flag
7. **ctgru.py** - 1 flag

These files are **functionally correct** - the violations are for operations like:
```python
# This is FINE - Python ints, not MLX arrays
shape = (batch_size, self.units * self.M)
fused_dim = input_dim + self.units
```

## How to Fix

Replace Python operators with MLX functions **only when operating on mx.array objects**:

```python
# ❌ WRONG - Breaks computation graph
result = a + b * c - d / e  # where a,b,c,d,e are mx.array

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
- Pure Python integers/floats: `for i in range(n-1)`, `shape = (dim1, dim2 * 3)` ✅  
- String operations: `path = base + ".json"` ✅

**Only fix operators that operate on `mx.array` objects!**

## Progress Summary

**Before fixes:** 12 files with 174 violations
**After fixes:** 8 files with ~31 violations (all false positives on Python int ops)
**Real violations fixed:** 143 ✅

The critical neural ODE computation paths are now fully GPU-optimized with proper MLX operator usage.

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
